"""Discord bot: posts in the text channel when an opted-in user says the trigger word.

    python -m triggerbot.bot          (needs DISCORD_TOKEN)

Uses models/detector.pt if it exists (`triggerbot train`), otherwise a demo detector that
fires on loud speech, so the whole pipeline can be tried before a real model exists.

Privacy by design:
- the bot only processes the voice of users who typed !optin (and !optout stops it);
- audio is kept in memory for about 1.5 s and never written to disk;
- the token comes from the environment, never from the code.

Threading: discord-ext-voice-recv calls `TriggerSink.write` from its own audio thread. That
method only appends bytes under a lock; an asyncio task picks the audio up every 250 ms and
runs the model in a worker thread, so the bot's event loop never blocks.
"""

import asyncio
import datetime
import os
import threading
from collections import Counter, defaultdict
from pathlib import Path

import discord
from discord.ext import commands, voice_recv

from .audio import pcm16_to_float, resample
from .detector import HOP_SECONDS, StreamingDetector
from .models import load_detector

DISCORD_RATE, DISCORD_CHANNELS = 48_000, 2
MODEL_PATH = Path(os.environ.get("TRIGGERBOT_MODEL", "models/detector.pt"))


class TriggerSink(voice_recv.AudioSink):
    def __init__(self, opted_in: set[int]):
        super().__init__()
        self.opted_in = opted_in
        self.pending: dict[int, list[bytes]] = defaultdict(list)
        self.lock = threading.Lock()

    def wants_opus(self) -> bool:
        return False  # we want decoded PCM

    def write(self, user, data) -> None:
        if user is None or user.id not in self.opted_in:
            return  # not consented: ignored, never buffered
        with self.lock:
            self.pending[user.id].append(data.pcm)

    def take(self) -> dict[int, bytes]:
        with self.lock:
            chunks = {uid: b"".join(parts) for uid, parts in self.pending.items()}
            self.pending.clear()
        return chunks

    def cleanup(self) -> None:
        with self.lock:
            self.pending.clear()


class TriggerBot(commands.Bot):
    def __init__(self, model):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        super().__init__(command_prefix="!", intents=intents)
        self.model = model
        self.opted_in: set[int] = set()
        self.detectors: dict[int, StreamingDetector] = {}
        self.counts: Counter[tuple[int, datetime.date]] = Counter()
        self.announce_to: discord.abc.Messageable | None = None
        self.sink: TriggerSink | None = None
        self.loop_task: asyncio.Task | None = None

    def detector_for(self, user_id: int) -> StreamingDetector:
        if user_id not in self.detectors:
            self.detectors[user_id] = StreamingDetector(self.model.predict_proba,
                                                        self.model.threshold)
        return self.detectors[user_id]

    async def process_audio(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await asyncio.sleep(HOP_SECONDS)
            if self.sink is None:
                continue
            for user_id, raw in self.sink.take().items():
                samples = resample(pcm16_to_float(raw, DISCORD_CHANNELS), DISCORD_RATE)
                hits = await loop.run_in_executor(None, self.detector_for(user_id).feed, samples)
                for _ in hits:
                    await self.announce(user_id)

    async def announce(self, user_id: int) -> None:
        key = (user_id, datetime.date.today())
        self.counts[key] += 1
        if self.announce_to is None:
            return
        user = self.get_user(user_id)
        name = user.display_name if user else f"<@{user_id}>"
        await self.announce_to.send(f"🔔 {name} said the word ({self.counts[key]} today)")


def build_bot(model) -> TriggerBot:
    bot = TriggerBot(model)

    @bot.event
    async def on_ready():
        print(f"Logged in as {bot.user}; detector: {getattr(model, 'name', 'CNN')}, "
              f"threshold {model.threshold:.2f}")

    @bot.command(help="Join your voice channel and start listening for opted-in users")
    async def join(ctx: commands.Context):
        if not ctx.author.voice:
            await ctx.send("Join a voice channel first.")
            return
        if ctx.voice_client:
            await ctx.voice_client.disconnect(force=True)
        vc = await ctx.author.voice.channel.connect(cls=voice_recv.VoiceRecvClient)
        bot.sink = TriggerSink(bot.opted_in)
        vc.listen(bot.sink)
        bot.announce_to = ctx.channel
        if bot.loop_task is None or bot.loop_task.done():
            bot.loop_task = asyncio.create_task(bot.process_audio())
        await ctx.send("Listening. Only users who type `!optin` are analysed; nothing is "
                       "recorded.")

    @bot.command(help="Leave the voice channel")
    async def leave(ctx: commands.Context):
        if ctx.voice_client:
            await ctx.voice_client.disconnect(force=True)
        bot.sink = None
        bot.detectors.clear()
        await ctx.send("Left the voice channel.")

    @bot.command(help="Let the bot analyse your voice for the trigger word")
    async def optin(ctx: commands.Context):
        bot.opted_in.add(ctx.author.id)
        await ctx.send(f"{ctx.author.display_name}: opted in. `!optout` to stop at any time.")

    @bot.command(help="Stop the bot analysing your voice")
    async def optout(ctx: commands.Context):
        bot.opted_in.discard(ctx.author.id)
        bot.detectors.pop(ctx.author.id, None)
        await ctx.send(f"{ctx.author.display_name}: opted out.")

    @bot.command(help="How many times each user said the word today")
    async def count(ctx: commands.Context):
        today = datetime.date.today()
        lines = [f"{(bot.get_user(uid) or ctx.author).display_name}: {n}"
                 for (uid, day), n in bot.counts.items() if day == today]
        await ctx.send("\n".join(lines) if lines else "Nobody said it today.")

    return bot


def main() -> None:
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        raise SystemExit("Set the DISCORD_TOKEN environment variable (see .env.example).")
    detector = load_detector(MODEL_PATH)
    if not MODEL_PATH.exists():
        print(f"{MODEL_PATH} not found: using the loudness demo detector "
              "(fires on loud speech, not on a word).")
    build_bot(detector).run(token)


if __name__ == "__main__":
    main()
