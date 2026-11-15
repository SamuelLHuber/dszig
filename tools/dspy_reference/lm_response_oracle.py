#!/usr/bin/env python3
"""Independent DSPy 3.5 LM response-normalization observations over HTTP.

Run with the reference virtualenv from the repository root. The localhost
fixture is deterministic protocol evidence, not authenticated model inference.
The native runtime neither imports nor executes this development oracle.
"""
import json,sys,threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
sys.path.insert(0,"tools/dspy_reference")
import dspy,file_oracle as fo
if dspy.__version__ != "3.5.0":
 raise RuntimeError("This oracle requires DSPy 3.5.0")
base={"id":"response-probe","model":"gpt-4o","output":[{"type":"message","content":[{"type":"output_text","text":"answer"}]}]}
cases=[]
def add(name,**changes):
 d=dict(base);d.update(changes);cases.append((name,d))
add("status_absent")
for state in ("completed","incomplete","failed"):
 add("status_"+state,status=state,incomplete_details={"reason":"max_output_tokens"})
add("nonassistant_incomplete_item",output=[{"type":"message","role":"user","status":"incomplete","content":[{"type":"output_text","text":"answer"}]}])
add("error_with_text",error={"code":"fixture_error","message":"source fixture error"})
add("output_null",output=None)
add("output_fallback",output=[],output_text="fallback")
add("refusal",output=[{"type":"message","content":[{"type":"refusal","refusal":"not allowed"}]}])
add("two_text_blocks",output=[{"type":"message","content":[{"type":"output_text","text":"first"},{"type":"text","text":"second"}]}])
add("tool_fallback_id",output=[{"type":"function_call","name":"probe","arguments":"{}"}])

add("empty_refusal",output=[{"type":"message","content":[{"type":"refusal","refusal":""}]}])
add("refusal_then_text",output=[{"type":"message","content":[{"type":"refusal","refusal":"no"},{"type":"output_text","text":"answer"}]}])
for value in (None,False,True,0,42,[1,"x"],{"x":1}):
 add("text_value_"+repr(value),output=[{"type":"message","content":[{"type":"text","text":value}]}])
for summary in ([],[{"type":"summary_text","text":""}],[{"text":"first"},{"text":"second"}],[{"text":None},"x",42],"visible"):
 add("reasoning_summary_"+repr(summary),output=[{"type":"reasoning","summary":summary}],output_text="must not override actual reasoning")
for arguments in (None,{},[1],"",'{ "query": "Paris", "limit": 2 }','[1,2]','{"bad":',42):
 add("tool_arguments_"+repr(arguments),output=[{"type":"function_call","name":"probe","arguments":arguments}])
add("tool_missing_arguments",output=[{"type":"function_call","name":"probe"}])
add("tool_empty_id",output=[{"type":"function_call","call_id":"","name":"probe","arguments":"{}"}])
add("tool_item_id",output=[{"type":"function_call","id":"item_id","name":"probe","arguments":"{}"}])
add("tool_after_text",output=[{"type":"message","content":[{"type":"output_text","text":"first"}]},{"type":"function_call","name":"probe","arguments":"{}"}])
add("tool_after_empty_reasoning",output=[{"type":"reasoning","summary":[]},{"type":"function_call","name":"probe","arguments":"{}"}])
add("ignored_output_entries",output=[None,17,{}, {"type":"unknown"}],output_text="fallback")


add("tool_unicode_object",output=[{"type":"function_call","name":"probe","arguments":{"é":"😀\n\x7f","quote":"\"\\"}}])
add("tool_unicode_json",output=[{"type":"function_call","name":"probe","arguments":json.dumps({"é":"😀\n\x7f","quote":"\"\\"},ensure_ascii=False)}])
add("tool_after_two_text_parts",output=[{"type":"message","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]},{"type":"function_call","name":"probe","arguments":"{}"}])

cases=[("responses",name,payload) for name,payload in cases]
for name,message in (
 ("chat_null_content",{"role":"assistant","content":None}),
 ("chat_missing_content",{"role":"assistant"}),
 ("chat_empty_content",{"role":"assistant","content":""}),
 ("chat_null_tool_content",{"role":"assistant","content":None,"tool_calls":[{"id":"call","type":"function","function":{"name":"probe","arguments":"{}"}}]}),
):
 cases.append(("chat",name,{"id":"chat-null-probe","model":"gpt-4o","choices":[{"message":message}]}))
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_POST(self):
  n=int(self.headers["Content-Length"]);self.rfile.read(n)
  payload=json.dumps(self.server.payload).encode();self.send_response(200);self.send_header("Content-Type","application/json");self.send_header("Content-Length",str(len(payload)));self.end_headers();self.wfile.write(payload)
server=ThreadingHTTPServer(("127.0.0.1",0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
rows=[]
try:
 for model_type,name,payload in cases:
  server.payload=payload
  model=dspy.LM("openai/gpt-4o",engine="lm15",model_type=model_type,api_base=f"http://127.0.0.1:{server.server_port}/v1",api_key="source-fixture-not-a-secret",cache=False,num_retries=0,timeout=2,max_tokens=None,temperature=None)
  rows.append({"id":name,"response":payload,"result":fo.observe(lambda:model(prompt="probe"))})
finally:
 server.shutdown();server.server_close();thread.join()
print(json.dumps({"dspy":dspy.__version__,"evidence_kind":"actual dspy.LM HTTP to localhost response fixture; not inference","cases":rows},ensure_ascii=False))
