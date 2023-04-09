import io
import json
import math
import os
import re

import discord
import nest_asyncio
import PyPDF2
import requests
import textwrap
from discord.ext import commands
from discord.ext.commands import ExpectedClosingQuoteError
from dotenv import load_dotenv

from langchain import OpenAI
from llama_index import (GPTTreeIndex, LLMPredictor, PromptHelper,
                         ServiceContext, StringIterableReader, download_loader)
from llama_index.readers import YoutubeTranscriptReader

nest_asyncio.apply()
load_dotenv()

intents = discord.Intents.all()
intents.members = True

bot = commands.Bot(command_prefix='!', intents=intents)
token = os.getenv('TOKEN')
api_token = str(os.getenv('OPENAI_API_KEY'))

max_input_size = 3600
num_output = 216
max_chunk_overlap = 512

prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name='gpt-3.5-turbo',
                                        max_tokens=num_output))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)


def is_valid_youtube_link(link):
    # Regular expression to match YouTube video URLs
    youtube_regex = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'

    return re.match(youtube_regex, link) is not None


def extract_text_from_pdf(file_content: bytes) -> str:
    with io.BytesIO(file_content) as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text


@bot.command(name='info')
async def info(ctx):
    embed = discord.Embed(title="Welcome to mayoBot",
                          description="The bot provides an interface for the GPT-3.5-turbo AI model over discord. "
                                      "The bot also allows for context building. "
                                      "You can upload a full PDF with 100s of pages, "
                                      "or link a 2+ hour youtube tutorial, and then query it using natural language.",
                          color=discord.Color.blue())
    embed.add_field(name="!info",
                    value="Displays this current message",
                    inline=False)
    embed.add_field(name="!ai <command>",
                    value="Queries the GPT-3.5-turbo AI model and returns the response",
                    inline=False)
    embed.add_field(name="!youtube <link>",
                    value="Indexes a youtube video's transcript",
                    inline=False)
    embed.add_field(name="!upload <PDF>",
                    value="Indexes the contents of the uploaded PDF",
                    inline=False)
    embed.add_field(name="!Q <command>",
                    value="Queries the provided index created by the !youtube or !upload command.",
                    inline=False)
    embed.add_field(name="Example Usage",
                    value="!youtube https://www.youtube.com/watch?v=y7iVTTH5tOA \n"
                          "!Q What are the key takeaways from this video? \n"
                          "!upload <upload pdf to discord> \n"
                          "!Q Summarize chapter 5 as if speaking to someone in middle school.\n"
                          "!ai Write a flask app using html js and css.",
                    inline=False)
    await ctx.send(embed=embed)


@bot.command(name='youtube')
async def youtube(ctx, link=None):
    global index
    YoutubeTranscriptReader = download_loader("YoutubeTranscriptReader")
    loader = YoutubeTranscriptReader()

    if link is None or not is_valid_youtube_link(link):
        embed = discord.Embed(title="Error",
                              description="Please provide a valid youtube link. \n"
                                          "For more information use the !info command.",
                              color=discord.Color.red())
        await ctx.send(embed=embed)
        return

    documents = loader.load_data(ytlinks=[link])
    index = GPTTreeIndex.from_documents(documents, service_context=service_context)
    index.save_to_disk('repo-youtube.json')

    await ctx.send(f"Video transcript processed. {ctx.author.mention}")


@bot.command(name='upload')
async def upload(ctx):
    if not ctx.message.attachments:
        embed = discord.Embed(title="Error",
                              description="Please upload a PDF file. \n"
                                          "For more information use the !info command.",
                              color=discord.Color.red())
        await ctx.send(embed=embed)
        return

    attachment = ctx.message.attachments[0]
    if not attachment.filename.endswith('.pdf'):
        embed = discord.Embed(title="Error",
                              description="Please upload a valid PDF file. \n"
                                          "For more information use the !info command.",
                              color=discord.Color.red())
        await ctx.send(embed=embed)
        return

    file_content = await attachment.read()
    file_content_str = extract_text_from_pdf(file_content)
    documents = StringIterableReader().load_data(texts=[file_content_str])
    global index
    index = GPTTreeIndex.from_documents(documents)
    index.save_to_disk('repo-string.json')

    await ctx.send(f"PDF file received and processed. {ctx.author.mention}")


@bot.command(name='Q')
async def query_index(ctx, *, command):
    global index
    prompt = "Limit your response to a maximum of 1024 characters. "
    try:
        response = index.query(prompt + command)
        sentences = str(response).split("\\n")
        embed = discord.Embed(color=discord.Color.blue())
        embed.set_thumbnail(url=ctx.author.avatar)
        embed.add_field(name="Query",
                        value=command,
                        inline=False,
                        )
        for sentence in sentences:
            print(sentence)
            if sentence:
                embed.add_field(name="Response",
                                value=sentence,
                                inline=False,
                                )
                await ctx.send(embed=embed)
                await ctx.send(f"Query Complete {ctx.author.mention}")
    except NameError:
        embed = discord.Embed(title="Error",
                              description="Please upload a PDF or youtube link first using the '!upload' or '!youtube' commands. \n"
                                          "For more information use the !info command.",
                              color=discord.Color.red())
        await ctx.send(embed=embed)


@bot.command(name='ai')
async def ai(ctx, *, command=None):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + api_token
    }
    data = json.dumps({
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": command}
        ],
        "max_tokens": 350,
    })
    url = "https://api.openai.com/v1/chat/completions"

    results = requests.post(url, headers=headers, data=data)

    ai_response = results.json()

    ai_response_clean = ai_response['choices'][0]['message']['content']
    sentences = ai_response_clean.split("\\n")

    embed = discord.Embed(color=discord.Color.blue())
    embed.set_thumbnail(url=ctx.author.avatar)
    embed.add_field(name="Query",
                    value=command,
                    inline=False,
                    )

    for sentence in sentences:
        print(sentence)
        if len(sentence) < 1000:
            embed.add_field(name="Response",
                            value=sentence,
                            inline=False,
                            )
            try:
                await ctx.send(embed=embed)
                await ctx.send(f"Query Complete {ctx.author.mention}")
            except ExpectedClosingQuoteError:
                embed = discord.Embed(title="Error",
                                      description="Error parsing prompt. Check for extra quotes",
                                      color=discord.Color.red())
                await ctx.send(embed=embed)
            except discord.errors.HTTPException:
                embed = discord.Embed(title="Error",
                                      description="Error parsing prompt. Prompt too large",
                                      color=discord.Color.red())
                await ctx.send(embed=embed)
        if len(sentence) > 999:
            response_count = 0
            chunk_size = int(len(sentence) / 4)
            sentence_chunks = textwrap.wrap(sentence, chunk_size)
            for chunk in sentence_chunks:
                response_count += 1
                if response_count > 4:
                    return
                large_embed = discord.Embed(title=f"Response Part {response_count}")
                large_embed.add_field(name="Response", value=chunk, inline=False)
                large_embed.set_thumbnail(url=ctx.author.avatar)
                try:
                    await ctx.send(embed=large_embed)
                except discord.errors.HTTPException:
                    embed = discord.Embed(title="Error",
                                          description="Response too large",
                                          color=discord.Color.red())
                    await ctx.send(embed=embed)
                finally:
                    if response_count == 4:
                        await ctx.send(f"Query Complete {ctx.author.mention}")

bot.run(token)
