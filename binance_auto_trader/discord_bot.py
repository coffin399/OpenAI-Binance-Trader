from __future__ import annotations

import asyncio
import logging
import threading
from typing import Iterable, Optional

import discord
from discord import app_commands
from discord.ext import commands

from binance_auto_trader.config.config import Config
from binance_auto_trader.main import TradingBot

logger = logging.getLogger(__name__)


class TradeControl(commands.Cog):
    def __init__(self, bot: commands.Bot, trading_bot: TradingBot) -> None:
        self.bot = bot
        self.trading_bot = trading_bot

    async def _check_permission(self, interaction: discord.Interaction) -> bool:
        if not hasattr(self.bot, "allowed_user_ids"):
            return True
        if interaction.user.id in self.bot.allowed_user_ids:
            return True
        await interaction.response.send_message("このコマンドは許可されたユーザーのみが実行できます。", ephemeral=True)
        return False

    @app_commands.command(
        name="stop-trade",
        description="全ポジションを決済し、トレードを一時停止します。",
    )
    async def stop_trade(self, interaction: discord.Interaction) -> None:
        if not await self._check_permission(interaction):
            return
        await interaction.response.defer(ephemeral=True)
        await asyncio.to_thread(self.trading_bot.pause_trading)
        await asyncio.to_thread(self.trading_bot.close_all_positions)
        await interaction.followup.send(
            "トレードを停止し、全てのポジションをクローズしました。",
            ephemeral=True,
        )

    @app_commands.command(name="start-trade", description="トレードを再開します。")
    async def start_trade(self, interaction: discord.Interaction) -> None:
        if not await self._check_permission(interaction):
            return
        await interaction.response.defer(ephemeral=True)
        await asyncio.to_thread(self.trading_bot.resume_trading)
        await interaction.followup.send("トレードを再開しました。", ephemeral=True)


class DiscordCommandBot(commands.Bot):
    def __init__(self, trading_bot: TradingBot, intents: discord.Intents, *, allowed_user_ids: set[int]) -> None:
        super().__init__(command_prefix="!", intents=intents)
        self.trading_bot = trading_bot
        self.allowed_user_ids = allowed_user_ids

    async def setup_hook(self) -> None:
        await self.add_cog(TradeControl(self, self.trading_bot))
        await self.tree.sync()

    async def on_ready(self) -> None:
        logger.info("Discord command bot ready as %s", self.user)


class DiscordRunner:
    def __init__(self, bot: commands.Bot, thread: threading.Thread) -> None:
        self._bot = bot
        self._thread = thread

    def stop(self) -> None:
        if self._bot.is_closed():
            return

        async def _close() -> None:
            await self._bot.close()

        try:
            asyncio.run_coroutine_threadsafe(_close(), self._bot.loop)
        except RuntimeError:
            threading.Thread(target=lambda: asyncio.run(_close()), daemon=True).start()

        self._thread.join(timeout=5)


def _parse_command_guilds(values: Optional[Iterable[str]]) -> list[discord.Object]:
    guild_ids: list[discord.Object] = []
    if not values:
        return guild_ids
    for value in values:
        try:
            guild_ids.append(discord.Object(id=int(value)))
        except (TypeError, ValueError):
            logger.warning("Invalid guild id '%s' in discord.command_guild_ids", value)
    return guild_ids


def start_discord_bot(config: Config, trading_bot: TradingBot) -> Optional[DiscordRunner]:
    discord_config = getattr(config, "discord", None)
    if not discord_config or not getattr(discord_config, "enabled", False):
        return None

    if not getattr(discord_config, "commands_enabled", False):
        logger.info("Discord commands disabled in configuration.")
        return None

    bot_token = getattr(discord_config, "bot_token", "")
    if not bot_token:
        logger.warning("Discord commands enabled but bot token is missing.")
        return None

    guild_values = getattr(discord_config, "command_guild_ids", [])
    guilds = _parse_command_guilds(guild_values)

    allowed_user_ids = set(getattr(discord_config, "allowed_user_ids", []))

    intents = discord.Intents.none()
    intents.guilds = True

    bot = DiscordCommandBot(trading_bot, intents=intents, allowed_user_ids=allowed_user_ids)

    async def sync_commands() -> None:
        if guilds:
            for guild in guilds:
                await bot.tree.sync(guild=guild)
        else:
            await bot.tree.sync()

    @bot.event
    async def setup_hook() -> None:  # type: ignore
        await bot.add_cog(TradeControl(bot, trading_bot))
        await sync_commands()

    def runner() -> None:
        try:
            bot.run(bot_token, log_handler=None)
        except Exception:  # noqa: BLE001
            logger.exception("Discord bot terminated unexpectedly")

    thread = threading.Thread(target=runner, name="DiscordBot", daemon=True)
    thread.start()

    return DiscordRunner(bot, thread)
