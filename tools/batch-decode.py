import os
import sys
import asyncio
import websockets
import json
import argparse

def parse_args():
  parser = argparse.ArgumentParser("A decoder websocket client capable of processing mutiple files")
  parser.add_argument("-u", "--url", default="ws://localhost:10086", type=str, help="server url")
  parser.add_argument("-o", "--outdir", default="", type=str, help="base directory")
  parser.add_argument("media", nargs="+", help="media files to serve")
  return parser.parse_args()

async def send_start(websocket):
  msg = '{"signal" : "start", "nbest" : 1, "continuous_decoding" : true}'
  await websocket.send(msg)

  repl = await websocket.recv()
  res = json.loads(repl)
  if res["status"] != "ok":
    print("Error starting the file: " + res["message"])
    exit(2)
  else:
    if res["type"] != "server_ready":
      print("Warning: received " + repl)

async def send_end(websocket, base_name, out_dir):
  msg = '{"signal" : "end"}'
  await websocket.send(msg)

  repl = await websocket.recv()
  res = json.loads(repl)
  if res["status"] != "ok":
    print("Error processing audio file: " + res["message"])
    exit(2)
  else:
    print("File processed: " + base_name)
    if res["type"] == "final_result":
      nbest = res["nbest"]
      print("nbest = " + nbest)
      out_name = os.path.join(out_dir, base_name + ".out")
      out_file = open(out_name, "w")
      out_file.write(nbest)
      out_file.close()

async def run_client(file_list, out_dir, uri):
  for file_name in file_list:
    print(f"Processing file: {file_name}")
    head, tail = os.path.split(file_name)

    async with websockets.connect(uri) as websocket:
      await send_start(websocket)

      CHUNK_SIZE = 4096
      f = open(file_name, "rb")
      chunk = f.read(CHUNK_SIZE)
      while len(chunk) != 0:
        await websocket.send(chunk)
        chunk = f.read(CHUNK_SIZE)
      f.close()
      await send_end(websocket, tail, out_dir)
      await websocket.close()

if __name__ == "__main__":
    args = parse_args()

    if args.outdir == "":
      print("--out-dir must be provided.")
      exit(1)

    if not os.path.isdir(args.outdir):
      print(f"--out-dir '{args.outdir}'' is not a directory.")
      exit(1)

    asyncio.get_event_loop().run_until_complete(run_client(args.media, args.outdir, args.url))
