from datetime import datetime
from pathlib import Path
import pickle
import os
import traceback

import discord
from discord.ext import commands

from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core import  StorageContext 
from llama_index.core import Settings
from llama_index.core.postprocessor import FixedRecencyPostprocessor
#from llama_index.embeddings import GeminiEmbedding
#from llama_index.core.embeddings import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.schema import TextNode, QueryBundle
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from llama_index.core import set_global_handler

import qdrant_client

import settings
from models import Message
from prompts import prompt

set_global_handler("simple")

logger = settings.logging.getLogger("bot")

persist_dir = "./.persist"

messages_path = Path(persist_dir + "/messages.pkl")
listening_path = Path(persist_dir + "/listening.pkl")

messages_path.parent.mkdir(parents=True, exist_ok=True)


def persist_listening():
  global listening

  # print(listening)
  with open(listening_path, 'wb') as file:
    pickle.dump(listening, file)


def persist_messages():
  global messages
  print("Persisting messages")
  # print(messages)
  with open(messages_path, 'wb') as file:
    pickle.dump(messages, file)


def process_incoming_message(message):
  """Replace user id with handle for mentions."""
  content = message.content
  for user in message.mentions:
    mention_str = f'<@{user.id}>'
    content = content.replace(mention_str, f'@{user.name}')
  message.content = content
  return message


if messages_path.is_file():
  with open(messages_path, 'rb') as file:
    messages = pickle.load(file)
else:
  messages: dict[int, list[Message]] = {} 
  persist_messages()

if listening_path.is_file():
  with open(listening_path, 'rb') as file:
    listening = pickle.load(file)
else:
  listening: dict[int, bool] = {}
  persist_listening()

# initialize qdrant client
qd_client = qdrant_client.QdrantClient(
  url=settings.QDRANT_URL,
  #api_key=settings.QDRANT_API_KEY
)

qd_collection = 'discord_llamabot_BAAI'

"""embed_model = OllamaEmbedding(
    model_name="llama3.2",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0, "temperature": 0.7},
)"""


from llama_index.embeddings.huggingface import HuggingFaceEmbedding
#from transformers import AutoTokenizer, AutoModel

# Load the Hugging Face model
model_name = "intfloat/multilingual-e5-large-instruct"
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModel.from_pretrained(model_name)

# Initialize the embedding provider
"""embed_model = HuggingFaceEmbedding(
    model=model,
    tokenizer=tokenizer,
    embed_batch_size=16  # Adjust batch size as needed
)"""
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
#from llama_index.embeddings import InstructorEmbedding
#embed_model = InstructorEmbedding(model_name="intfloat/multilingual-e5-large-instruct")

#intfloat/multilingual-e5-large-instruct

use_openai = bool(os.environ.get("USE_OPENAI", False))
use_cohere = bool(os.environ.get("USE_COHERE", False))
use_ollama = bool(os.environ.get("USE_OLLAMA", False))
# print(use_openai)

# if os.environ.get("GOOGLE_API_KEY", None):
if use_openai:
  from llama_index.llms import OpenAI
  # from llama_index.embeddings import OpenAIEmbedding
  print("Using GPT-4")
  llm=OpenAI(
    model="gpt-4-0125-preview",
  )
  # embed_model = OpenAIEmbedding(model="text-embedding-3-small")
elif use_cohere:
  from llama_index.llms import Cohere
  print("Using Cohere")
  llm=Cohere(api_key=os.environ.get('COHERE_KEY'))
elif use_ollama:
  from llama_index.llms.ollama import Ollama
  print("Using Llama 3.2 via Ollama")
  llm = Ollama(
    model="llama3.2",
    base_url="http://localhost:11434",  # Default Ollama server URL
  )
else:
  from llama_index.llms import Gemini
  print("Using Gemini Pro")
  llm=Gemini()

vector_store = QdrantVectorStore(client=qd_client,
                                 collection_name=qd_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
Settings.llm = llm
Settings.embed_model = embed_model

index = VectorStoreIndex([],
               storage_context=storage_context,
               embed_model=embed_model)

persist_messages()
persist_listening()

def run():
  "Run LlamaBot."
  intents = discord.Intents.default()
  intents.message_content = True

  bot = commands.Bot(
    command_prefix='/',
    intents=intents
  )


  @bot.event
  async def on_ready():
    logger.info(f"User: {bot.user} (ID: {bot.user.id})")

  @bot.command(
    aliases=['li']
  )
  async def listen(ctx):
      "Llama will start listening to messages in this channel from now on."

      global listening
      listening[ctx.guild.id] = True
      persist_listening()
      logger.info(f"Listening to messages on channel {ctx.channel.name} of server: {ctx.guild.id} "
                f"from {datetime.now().strftime('%m-%d-%Y %H:%M:%S')}")
      await ctx.send('Listening to your messages now.')
  @bot.command()
  async def read_history(ctx, channel_id: int = None):
      """Command to read the message history of a specific channel and persist it."""
      channel = bot.get_channel(ctx.channel.id if channel_id is None else channel_id)
      await ctx.send(f"Reading history for channel: {channel.name}")

      async for message in channel.history(limit=None):  # Adjust the limit as needed
          when = message.created_at
          who = message.author
          msg_content = message.content

          # Process message content for mentions
          processed_message = process_incoming_message(message)

          # Log and persist the message
          msg_str = f"[{when.strftime('%m-%d-%Y %H:%M:%S')}] - @{who} on #[{str(message.channel)[:15]}]: `{processed_message.content}`"

          if not messages.get(message.guild.id, None):
              messages[message.guild.id] = []

          messages[message.guild.id].append(Message(
              is_in_thread=str(message.channel.type) == 'public_thread',
              posted_at=when,
              author=str(who),
              message_str=msg_str,
              channel_id=message.channel.id,
              just_msg=processed_message.content
          ))
        # Create a TextNode for Qdrant
          node = TextNode(
              text=msg_str,
              metadata={
                  'author': str(who),
                  'posted_at': str(when),
                  'channel_id': message.channel.id,
                  'guild_id': message.guild.id
              },
              excluded_llm_metadata_keys=['author', 'posted_at', 'channel_id', 'guild_id'],
              excluded_embed_metadata_keys=['author', 'posted_at', 'channel_id', 'guild_id'],
          )

          # Insert the node into Qdrant
          index.insert_nodes([node])
      print("Haciendo persistentes {} messages".format(len(messages[message.guild.id])))
      # Persist the updated messages
      persist_messages()
      await ctx.send("History reading and persistence complete.")

  @bot.command(
    aliases=['s']
  )
  async def stop(ctx):
      "Llama will stop listening to messages in this channel from now on."

      global listening
      listening[ctx.guild.id] = False
      persist_listening()
      logger.info(f"Stopped Listening to messages on channel "
              f"{ctx.channel.name} from {datetime.now().strftime('%m-%d-%Y')}")
      await ctx.send('Stopped listening to messages.')

  @bot.command(
    aliases=['f']
  )
  async def forget(ctx):
      "Llama will forget everything it remembered. It will forget all messages, todo, reminders etc."

      forget_all(ctx)
      await ctx.send('All messages forgotten & stopped listening to yall')

  @bot.command(
    aliases=['st']
  )
  async def status(ctx):
    "Status of LlamaBot, whether it's listening or not"

    await ctx.send(
      "Listening to yall👂" if listening.get(ctx.guild.id, False) \
      else "Not Listening 🙉"
    )

  @bot.command(
    aliases=['l']
  )
  async def llama(ctx, *query):
      "Llama will answer your query"

      global listening
    
      if not listening.get(ctx.guild.id, False):
        await ctx.message.reply(
          "I'm not listening to what y'all saying 🙈🙉🙊. "
          "\nRun \"/listen\" if you want me to start listening."
        )
        return

      if len(query) == 0:
        await ctx.message.reply("What?")
        return
      user_messages = [msg for msg in messages.get(ctx.guild.id, []) if msg.author!=str(bot.user) and not msg.just_msg.startswith("/")]
      # print(user_messages)
      if len(user_messages) == 0:
        await ctx.message.reply("Hey, Bot's knowledge base is empty now. Please say something before asking it questions.")
        return
        
      try:
        async with ctx.typing():
          await ctx.message.reply(await answer_query(" ".join(query), ctx, bot))
      except:
        tb = traceback.format_exc()
        print(tb)
        await ctx.message.reply("The bot encountered an error, will try to fix it soon. Feel free to send a dm to @rsrohan99 about it or open an issue on GitHub https://github.com/rsrohan99/llamabot, any kind of feedback is really appreciated, thanks.")

  @bot.event
  async def on_message(message):
      global listening
      # if message.author == bot.user:
      #     return
      message = process_incoming_message(message)

      if listening.get(message.guild.id, False):
        if message.content.startswith('/'):
          if message.content.startswith('/l') or message.content.startswith('/llama'):
            remember_message(message, True)
        else:
          remember_message(message, message.author==bot.user)


      await bot.process_commands(message)

  def remember_message(message, save_only_message):
    when = message.created_at
    who=message.author
    msg_content = message.content

    # print(message)

    logger.info(
    f"Remembering new message \"{msg_content}\" from {who} on channel "
    f"{message.channel.name} at {datetime.now().strftime('%m-%d-%Y %H:%M:%S')}"
    )

    msg_str = f"@{who} on #[{str(message.channel)[:15]}]: `{msg_content}`"

    if not save_only_message:
      node = TextNode(
        text=msg_str,
        metadata={
          'author': str(who),
          'posted_at': str(when),
          'channel_id': message.channel.id,
          'guild_id': message.guild.id
        },
        excluded_llm_metadata_keys=['author', 'posted_at', 'channel_id', 'guild_id'],
        excluded_embed_metadata_keys=['author', 'posted_at', 'channel_id', 'guild_id'],
      )

      index.insert_nodes([node])

    if not messages.get(message.guild.id, None):
      messages[message.guild.id] = []
    messages[message.guild.id].append(Message(
      is_in_thread=str(message.channel.type) == 'public_thread',
      posted_at=when,
      author=str(who),
      message_str=msg_str,
      channel_id=message.channel.id,
      just_msg=message.content
    ))
    persist_messages()

  async def answer_query(query, ctx, bot):
    channel_id = ctx.channel.id
    thread_messages = [
      msg.message_str for msg in messages.get(ctx.guild.id, []) if msg.channel_id==channel_id
    ][-1*settings.LAST_N_MESSAGES:-1]
    partially_formatted_prompt = prompt.partial_format(
      replies="\n".join(thread_messages),
      user_asking=str(ctx.author),
      bot_name=str(bot.user)
    )

    filters = MetadataFilters(
      filters=[
        MetadataFilter(
          key="guild_id", operator=FilterOperator.EQ, value=ctx.guild.id
        ),
        MetadataFilter(
          key="author", operator=FilterOperator.NE, value=str(bot.user)
        ),
      ]
    )
    
    postprocessor = FixedRecencyPostprocessor(
        top_k=2, 
        date_key="posted_at", # the key in the metadata to find the date
    )
    query_engine = index.as_query_engine(
      filters=filters,
      similarity_top_k=30,
      #node_postprocessors=[postprocessor]
      )
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": partially_formatted_prompt}
    )

    replies_query = [
      msg.just_msg for msg in messages.get(ctx.guild.id, []) if msg.channel_id==channel_id
    ][-1*settings.LAST_N_MESSAGES:-1]
    replies_query.append(query)

    print(replies_query)
    return query_engine.query(QueryBundle(
      query_str=query,
      custom_embedding_strs=replies_query
    ))


  def forget_all(ctx):
    from qdrant_client.http import models as rest

    global qd_client

    try:
      messages.pop(ctx.guild.id)
      listening.pop(ctx.guild.id)
    except KeyError:
      pass
    persist_messages()
    persist_listening()


    qd_client.delete(
        collection_name=qd_collection,
        points_selector=rest.Filter(
          must=[
            rest.FieldCondition(
                key="guild_id", match=rest.MatchValue(value=ctx.guild.id)
            )
          ]
        ),
    )


  bot.run(settings.DISCORD_API_SECRET, root_logger=True)


if __name__ == "__main__":
    run()
