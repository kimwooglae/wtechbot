from discord_slash import SlashContext, cog_ext
from discord.ext import commands

class PingCommand(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @cog_ext.cog_slash(name="ping", description="Replies with Pong!")
    async def ping(self, ctx: SlashContext):
        await ctx.send(content="Pong!")

def setup(bot):
    bot.add_cog(PingCommand(bot))
