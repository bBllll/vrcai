"use strict";
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="session-token"]').content;
let config = null, state = {state:"stopped"}, cursor = 0, pttTimer = null, pollTimer = null;
const fieldNodes = new Map();
async function api(path, data) {
  const options = {headers:{"X-Session-Token":token}};
  if (data !== undefined) {
    options.method = "POST"; options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(data);
  }
  const response = await fetch("/api/" + path, options);
  const result = await response.json();
  if (!response.ok || result.ok === false) throw new Error(result.error || "请求失败");
  return result;
}
function notice(text, error=false) {
  $("notice").hidden = !text; $("notice").textContent = text; $("notice").className = error ? "error" : "";
}
function report(error) { notice(error.message || String(error), true); }
function valueAt(object, path) { return path.split(".").reduce((v,k)=>v?.[k], object); }
function putAt(object, path, value) {
  const parts = path.split("."); let target = object;
  parts.slice(0,-1).forEach(k => {target[k] ||= {}; target=target[k];});
  target[parts.at(-1)] = value;
}
function addLog(text) {
  const box=$("logs"), pinned=box.scrollTop + box.clientHeight >= box.scrollHeight-40;
  const line=document.createElement("div"); line.textContent=text;
  line.className=/WARNING|ERROR/.test(text)?"log-error":text.includes("回复:")?"log-reply":"";
  box.append(line); while(box.children.length>500)box.firstChild.remove();
  if(pinned)box.scrollTop=box.scrollHeight;
}
function updateStatus(next) {
  state=next;
  const busy=["starting","stopping"].includes(next.state), running=next.state==="running";
  const names={starting:"正在启动…",running:"正在监听",stopping:"正在停止，等待当前任务结束…",stopped:"已停止",error:"启动或设备出错"};
  $("status").textContent=names[next.state]||"状态未知"; $("status").dataset.state=next.state;
  $("start").hidden=running||busy; $("stop").hidden=!(running||busy); $("stop").disabled=next.state==="stopping";
  for(const id of ["interrupt","reset","send"])$(id).disabled=!running;
  $("ptt").disabled=!running||next.listen_mode!=="push_to_talk";
  if(!running)endPtt(false);
  $("device").textContent=next.audio_device ? "输入："+next.audio_device+" · 电平 "+next.audio_rms : "请先完成设置并开启对话。";
  if(running)$("tts-toggle").checked=next.tts_enabled;
  const m=next.metrics||{};
  for(const [id,key] of [["stt-ms","stt_ms"],["first-ms","llm_first_token_ms"],["llm-ms","llm_ms"],["tts-ms","tts_ms"]])
    $(id).textContent=m[key]==null?"—":m[key]+" ms";
  $("queue-state").textContent=running ? "等待识别/回复 "+(next.input_queue||0)+" · 等待播放 "+(next.tts_queue||0)+" · 丢弃旧输入 "+(m.dropped_inputs||0)+" · 过期 "+(m.stale_inputs||0) : "等待启动";
  if(next.error)notice(next.error,true);
  else if(next.restart_required)notice("设置已保存。停止对话后重新开启，使新配置生效。");
}
async function poll() {
  try {
    const [s, logs]=await Promise.all([api("status"),api("logs?after="+cursor)]);
    updateStatus(s);
    for(const entry of logs.logs)addLog(entry.text);
    cursor=logs.cursor;
  } catch(e) { $("status").textContent="管理服务连接中断"; report(e); }
  finally { pollTimer=setTimeout(poll,800); }
}
async function action(path, body={}) {
  try {const result=await api(path,body);if(result.state)updateStatus(result);return result;}
  catch(e){report(e);return null;}
}
const fields = [
 ["人设与对话", [
  ["system_prompt","角色设定","textarea"],["chat.reply_prefix","聊天框回复前缀"],
  ["chat.trigger_prefix","语音唤醒前缀（留空表示不限制）"],
  ["chat.listen_mode","监听方式","select",["continuous","push_to_talk"]],
  ["chat.temperature","回复随机程度（0～2）","number",0,2,.1],
  ["chat.max_tokens","回复最大 Token","number",1,8192,1],
  ["chat.max_history","上下文消息数（按完整问答保留）","number",0,200,1],
  ["streaming.enabled","边生成边回复","checkbox"],
  ["chat.queue_size","待处理输入上限","number",1,10,1],
  ["chat.max_input_age","输入超过多少秒后丢弃","number",1,120,1]]],
 ["语音识别", [
  ["stt.provider","识别服务","select",["groq_whisper","openai_whisper","dashscope_qwen_asr","funasr"]],
  ["stt.language","识别语言（留空自动，例如 zh / en / ja）"],
  ...["groq_whisper","openai_whisper","dashscope_qwen_asr"].flatMap(p=>[
   ["stt."+p+".api_key",p+" API Key","password"],["stt."+p+".model",p+" 模型"]]),
  ["stt.openai_whisper.base_url","OpenAI Whisper Base URL（空为官方默认）"],
  ["stt.dashscope_qwen_asr.base_url","Qwen ASR Base URL"],
  ["stt.funasr.model","FunASR 模型"],["stt.funasr.device","FunASR 设备（cpu / cuda:0）"]]],
 ["对话模型",[
  ["llm.provider","对话服务","select",["openai_compatible","openai","ollama"]],
  ...["openai_compatible","openai","ollama"].flatMap(p=>[
   ...(p!=="ollama"?[["llm."+p+".api_key",p+" API Key","password"]]:[]),
   ["llm."+p+".base_url",p+" Base URL"],["llm."+p+".model",p+" 模型"]])]],
 ["语音回复",[
  ["tts.enabled","开启语音回复","checkbox"],["tts.minimax.api_key","MiniMax API Key","password"],
  ["tts.minimax.base_url","MiniMax Base URL"],["tts.minimax.model","MiniMax 模型"],
  ["tts.minimax.voice_id","音色 voice_id"],
  ["tts.minimax.speed","语速","number",.5,2,.1],["tts.minimax.vol","音量","number",.1,10,.1],
  ["tts.output_device","播放设备","device"],["tts.mute_capture","AI 播放时暂停识别，避免回声","checkbox"]]],
 ["音频与分段",[
  ["audio.device","输入设备（自动选择系统声音）","device"],
  ["audio.sample_rate","识别采样率","number",8000,48000,1],
  ["audio.chunk_duration","音频帧时长（秒）","number",.01,.1,.01],
  ["audio.silence_duration","停顿多久提交语音（秒）","number",.1,3,.1],
  ["audio.min_speech_duration","最短有效语音（秒）","number",.04,2,.02],
  ["audio.max_speech_duration","最长片段（秒）","number",2,30,1],
  ["audio.energy_threshold","最低语音能量阈值","number",1,10000,1],
  ["audio.noise_ratio","高于噪声的倍数","number",1.1,10,.1],
  ["audio.pre_roll_duration","保留开头音频（秒）","number",0,.5,.02]]],
 ["连接与聊天框",[
  ["osc.host","VRChat OSC 地址"],["osc.vrc_port","OSC 端口","number",1,65535,1],
  ["osc.max_chars","每条消息字符上限","number",16,144,1],
  ["osc.send_interval","聊天框发送间隔（秒）","number",1,10,.1],
  ["network.timeout","网络请求超时（秒）","number",2,120,1],
  ["network.max_retries","网络自动重试次数","number",0,2,1]]]
];
function renderSettings() {
  $("settings-form").replaceChildren();fieldNodes.clear();
  for(const [title, items] of fields) {
    const details=document.createElement("details");details.open=title==="人设与对话";
    const summary=document.createElement("summary");summary.textContent=title;details.append(summary);
    const grid=document.createElement("div");grid.className="fields";details.append(grid);
    for(const [path,label,type="text",lo,hi,step] of items) {
      const row=document.createElement("div");row.className="field"+(type==="textarea"?" wide":"");
      const caption=document.createElement("label");caption.htmlFor="field-"+path;caption.textContent=label;row.append(caption);
      const input=document.createElement(type==="textarea"?"textarea":type==="select"||type==="device"?"select":"input");
      input.id="field-"+path;
      if(input.tagName==="INPUT")input.type=type;
      if(type==="textarea")input.rows=5;
      if(type==="number"){input.min=lo;input.max=hi;input.step=step;}
      if(type==="select")for(const option of lo){const node=document.createElement("option");node.value=option;node.textContent=({continuous:"自动监听",push_to_talk:"按住按钮 / 空格监听"})[option]||option;input.append(node);}
      const original=valueAt(config,path);
      if(type==="device"){
        const option=document.createElement("option");option.value="";option.textContent=path==="audio.device"?"自动：默认系统声音":"默认播放设备";input.append(option);
        if(original!=null){const option=document.createElement("option");option.value=original;option.textContent="设备 "+original+"（刷新查看名称）";input.append(option);}
      }
      if(type==="checkbox")input.checked=original;
      else input.value=original??"";
      if(type==="password"){
        input.value="";input.autocomplete="new-password";
        input.placeholder=valueAt(config,path.replace(/api_key$/,"key_configured"))?"已配置，留空保留":"尚未配置";
        const group=document.createElement("div");group.className="secretrow";group.append(input);
        const clear=document.createElement("button");clear.type="button";clear.textContent="清除";
        clear.addEventListener("click",()=>{input.value="";input.dataset.clear="true";input.placeholder="保存时清除原密钥";});
        input.addEventListener("input",()=>delete input.dataset.clear);group.append(clear);row.append(group);
      }else row.append(input);
      const parts=path.split(".");
      if((parts[0]==="llm"||parts[0]==="stt")&&parts.length===3){row.dataset.providerSection=parts[0];row.dataset.provider=parts[1];}
      fieldNodes.set(path,{input,type,original,row});grid.append(row);
      if(path==="llm.provider"||path==="stt.provider")input.addEventListener("change",updateProviderFields);
    }
    $("settings-form").append(details);
  }
  updateProviderFields();
}
function updateProviderFields(){
  for(const section of ["stt","llm"]){
    const selected=fieldNodes.get(section+".provider").input.value;
    document.querySelectorAll("[data-provider-section='"+section+"']").forEach(row=>row.hidden=row.dataset.provider!==selected);
  }
}
async function openSettings() {
  try{config=await api("config");renderSettings();$("settings-error").textContent="";$("settings").showModal();}
  catch(e){report(e);}
}
async function saveSettings(){
  const updates={};
  for(const [path,{input,type,original}] of fieldNodes){
    if(type==="password"){
      if(input.dataset.clear==="true")putAt(updates,path,null);
      else if(input.value)putAt(updates,path,input.value);
      continue;
    }
    let value=type==="checkbox"?input.checked:type==="device"?(input.value===""?null:Number(input.value)):type==="number"?Number(input.value):input.value;
    if(type==="number"&&(!input.value.trim()||!input.checkValidity())){$("settings-error").textContent="请检查："+input.labels[0].textContent;input.focus();return;}
    if(value!==original)putAt(updates,path,value);
  }
  $("settings-save").disabled=true;
  try{
    const result=await api("config",updates);
    config=await api("config");$("tts-toggle").checked=config.tts.enabled;
    $("settings").close();notice(result.restart_required?"设置已保存。请停止对话后重新开启，使新配置生效。":"设置已保存。");
  }catch(e){$("settings-error").textContent=e.message;}
  finally{$("settings-save").disabled=false;}
}
async function refreshDevices(){
  $("refresh-devices").disabled=true;
  try{
    const result=await api("devices");
    for(const path of ["audio.device","tts.output_device"]){
      const node=fieldNodes.get(path).input, selected=node.value;
      while(node.options.length>1)node.remove(1);
      for(const d of result.devices){
        if(path==="audio.device"?!d.inputs&&!d.loopback:!d.outputs)continue;
        const option=document.createElement("option");option.value=d.id;option.textContent="["+d.id+"] "+d.name+" · "+d.api;node.append(option);
      }
      if(selected&&!Array.from(node.options).some(o=>o.value===selected)){
        const option=document.createElement("option");option.value=selected;option.textContent="设备 "+selected+"（当前不可用）";node.append(option);
      }
      node.value=selected;
    }
    $("settings-error").textContent="";
  }catch(e){$("settings-error").textContent=e.message;}
  finally{$("refresh-devices").disabled=false;}
}
async function beginPtt(){
  if(pttTimer||$("ptt").disabled)return;
  $("ptt").classList.add("active");
  pttTimer=setInterval(()=>api("listen",{active:true}).catch(report),400);
  try{await api("listen",{active:true});}catch(e){endPtt(false);report(e);}
}
function endPtt(send=true){
  if(!pttTimer)return;
  clearInterval(pttTimer);pttTimer=null;$("ptt").classList.remove("active");
  if(send)api("listen",{active:false}).catch(report);
}
$("start").addEventListener("click",async()=>{$("start").disabled=true;notice("");await action("start");$("start").disabled=false;});
$("stop").addEventListener("click",()=>action("stop"));
$("interrupt").addEventListener("click",()=>action("interrupt"));
$("reset").addEventListener("click",()=>action("reset"));
$("tts-toggle").addEventListener("change",async()=>{
  const enabled=$("tts-toggle").checked;
  const result=await action("config",{tts:{enabled}});
  if(!result)$("tts-toggle").checked=!enabled;
});
$("clear-logs").addEventListener("click",()=>$("logs").replaceChildren());
$("settings-open").addEventListener("click",openSettings);
$("settings-close").addEventListener("click",()=>$("settings").close());
$("settings-save").addEventListener("click",saveSettings);
$("settings-form").addEventListener("submit",e=>e.preventDefault());
$("refresh-devices").addEventListener("click",refreshDevices);
$("diagnose").addEventListener("click",async()=>{
  try{
    const result=await api("diagnostics");$("checks").replaceChildren();
    for(const check of result.checks){
      const li=document.createElement("li"),title=document.createElement("b");
      title.textContent=(check.ok?"✓ ":"✗ ")+check.name;li.append(title,document.createTextNode(check.detail));$("checks").append(li);
    }
  }catch(e){report(e);}
});
$("test-osc").addEventListener("click",async()=>{const r=await action("test-osc");if(r)notice(r.message);});
$("message-form").addEventListener("submit",async e=>{
  e.preventDefault();const text=$("message").value.trim();if(!text)return;
  $("send").disabled=true;const result=await action("message",{text});
  if(result)$("message").value="";$("send").disabled=state.state!=="running";
});
$("ptt").addEventListener("pointerdown",e=>{e.preventDefault();$("ptt").setPointerCapture(e.pointerId);beginPtt();});
for(const event of ["pointerup","pointercancel","lostpointercapture"])$("ptt").addEventListener(event,()=>endPtt());
document.addEventListener("keydown",e=>{
  if(e.code==="Space"&&!e.repeat&&!$("settings").open&&!/INPUT|TEXTAREA|SELECT|BUTTON/.test(e.target.tagName)&&!e.target.isContentEditable&&!$("ptt").disabled){e.preventDefault();beginPtt();}
});
document.addEventListener("keyup",e=>{if(e.code==="Space")endPtt();});
window.addEventListener("blur",()=>endPtt());
document.addEventListener("visibilitychange",()=>{if(document.hidden)endPtt();});
(async()=>{
  try{config=await api("config");$("tts-toggle").checked=config.tts.enabled;}
  catch(e){report(e);}
  poll();
})();
