# 实时视频拍摄 Agent：骨架

按《设计文档》第 3 节搭的第一版：目标是整条 agent loop 能转起来，功能不求全。

现在能做的：浏览器每秒送 3 帧 → 快回路给取景提示 → 手动或计时结束一条片段 → 慢回路结算 →
账本更新 → 通过就进下一个镜头，不通过原位重拍。自带 3 个手写镜头，围绕"桌下的包"一个例子。
配了模型的密钥之后，可以在界面里写一句想法或贴一段脚本，让模型拆成镜头清单（见"换一个脚本"）。
语义判断默认是 mock；真实的 Cosmos 客户端按主办方文档写好了，还没连过真实端点。

## 跑起来

```bash
python3 -m venv .venv && source .venv/bin/activate    # Windows：.venv\Scripts\activate
pip install -r requirements.txt     # ultralytics 体积大，可以先不装（见下）
python run.py                       # 打开 http://localhost:8000
```

- **没有摄像头、没装 YOLO 也能跑**：页面右上角"视频源"切到"模拟画面"，拖滑块摆桌子、包和人，
  整条回路都是真的在转（只是把 YOLO 换成了按颜色认色块）。
- 没装 ultralytics 时用摄像头：靠检测的约束自动降级成文字提示，界面右上角会说明原因；清单照样能走完。
- 在终端里看一遍讲解版：`python scripts/simulate.py`
- 测试：`pytest`（194 个；其中 4 个要 ultralytics，没装会自动跳过）

常用参数：`--detector yolo|none`、`--judge cosmos`、`--take-seconds 8`（一个人又拍又演时把片段调长）、
`--yolo-device mps`（Apple 芯片）、`--fresh`（清空进度）。全部可调的阈值在 `shotagent/config.py`。

## 换一个脚本：想法 → 镜头清单

不想拍希区柯克，就让模型按你的想法或脚本出清单：

```bash
export OPENAI_API_KEY=sk-...        # 默认连 OpenAI，模型 gpt-5.6-luna
python scripts/try_plan.py "深夜加班的人去倒了杯水，回来发现桌上多了一本不是自己的书"   # 先单独试一次
python run.py                       # 然后在界面右边点"换一个脚本"
```

- 输入可以是一句想法、一种风格，也可以是一段写好的脚本（脚本会按原样拆，不添情节）。
- 模型拿到的是一张**菜单**：系统看得见哪些角色（人、包、桌子、椅子、杯子……见 `config.DEFAULT_LABEL_MAP`）、
  能自动检查哪些关系（`constraints.PREDICATES` 里登记的那几种）。菜单之外的要求，它得放进"拍完由语义模型判断"
  或"只给文字提示"——界面上每条约束后面标着由谁检查，和手写清单一样。
- 模型的输出不直接用：谓词名、角色名逐个核对，ID 和提示措辞由程序填。校验不过会带着原因让它重写一次，
  再不行就报错，账本原样不动。
- 用"模拟画面"时只让模型用人、包、桌子（模拟画面里只画得出这三样），所以生成的清单在模拟里也能拍完。
- 换清单会清空进度；"从头开始"只清进度、不换清单，也不会再问一次模型。"换回示例清单"回到手写的那 3 个镜头。
- 到现场改用主办方的模型：设 `SHOTAGENT_LLM_BASE_URL`、`SHOTAGENT_LLM_API_KEY`、`SHOTAGENT_LLM_MODEL` 三个环境变量
  （需要对方是 OpenAI 兼容的 chat/completions 接口）。
- 没有密钥也想走一遍这条路：`python scripts/fake_endpoints.py`，再
  `OPENAI_BASE_URL=http://127.0.0.1:9001/v1 OPENAI_API_KEY=fake python run.py`，回来的是一份写死的清单。

## 模块关系

一句话：**两条回路共用一本账本，但能写它的只有规划层和慢回路**。快回路在每个镜头开始时拿一份快照，之后只读；
`session.py` 是把它们串起来的状态机，自己也不写账本。

### 谁调用谁

```
  web/  (浏览器)                                                  按钮：开拍 / 结束 / 重拍 / 从头开始
  sources.js --一帧 JPEG + 时间戳 + 能力说明--> server.py  <--------------+
                                                    |
                                             frame_source.py      -> Frame
                                                    |
                                               session.py         状态机：取景 -> 拍摄 -> 检查
          +-------------------+---------------------+-------+---------------------+
          | 每一帧            | 拍摄中的每一帧              | 拍完一条            | 启动 / 重置
          v                   v                             v                     v
    fast_loop.py          takes.py                    slow_loop.py           planner.py
     |        |         (攒帧、落盘)                   |        |              |       |
perception/  constraints.py                         judge/   constraints.py  plans.py  plan_gen.py
 yolo        spatial.py                              mock    spatial.py      (手写的    (想法 / 脚本 -> 清单，
 synthetic                                           cosmos                   示例)      模型由 llm.py 去问)
 null
          :                                                 |                     |
          : 只拿快照（每个镜头开始时一次）                  | commit()            | commit()
          :                                                 v                     |
          +. . . . . . . . . . . . . . . . . . . . . .>  ledger/  <---------------+
                                                         store.py    单一写入出口
                                                         reducer.py  规则：有效素材、字段信谁
                                                         state.py    镜头日志 + 场景状态
```

实线是调用，点线是"只给快照"。`constraints.py` / `spatial.py` 两条回路都用：快回路拿它们逐帧判断，
慢回路拿同一套换算把片段里的 YOLO 观测汇总成场景事实——写进账本的和读出来的是同一把尺子。

### 每个文件管什么

| 文件 | 职责 | 读账本 | 写账本 |
|---|---|---|---|
| `contracts.py` | 模块之间传的数据结构（帧、检测、约束、提示、片段、判断） | | |
| `config.py`、`texts.py`、`media.py` | 可调参数；通用措辞；图像和视频编解码 | | |
| `frame_source.py` + `web/js/sources.js` | 视频源 → 一帧图片 + 时间戳 + 能力说明 | | |
| `planner.py` + `plans.py` | 指令 → 镜头清单：空指令给手写的 3 个镜头，否则交给生成器 | 是 | 镜头清单 |
| `plan_gen.py` + `llm.py` | 生成器：想法 / 脚本 → 清单（菜单、校验、重试）；OpenAI 兼容的模型客户端 | | |
| `fast_loop.py` | 一帧 → 取景提示 | 只拿快照 | 不能 |
| `constraints.py` + `spatial.py` | 关系谓词；相对位置和朝向的换算 | | |
| `perception/` | 检测器：YOLO / 模拟画面 / 空 | | |
| `slow_loop.py` | 一条片段 → 结论 + 场景事实 | 是 | 片段结算、场景事实 |
| `judge/` | 语义判断：mock / Cosmos | | |
| `takes.py` | 片段攒帧、落盘 | | |
| `ledger/` | 账本：状态、事件、规则、写入出口 | | |
| `session.py` | agent loop 的状态机，把上面的串起来 | 只读函数 | 不能（只能请规划层重排、请慢回路结算） |
| `wiring.py` | 所有对象在这里创建和连接 | | |
| `server.py` | HTTP 接口，薄壳 | | |

### 模块之间传的是什么

除了账本自己的两种（事件、状态），其余都定义在 `contracts.py`。

| 数据 | 谁产生 | 谁用 |
|---|---|---|
| `SourceCaps` 视频源能力说明 | 浏览器登记（`/api/source`） | `fast_loop.load`：决定哪些约束能自动检查、哪些降级 |
| `Frame` 一帧 | `frame_source.ingest` | `session.on_frame` → `fast_loop.step` |
| `Observation` 这一帧看到了什么 | `fast_loop.step`（检测器的输出 + 传感器读数） | 随 `Guidance` 返回；拍摄时存进片段；慢回路从中汇总位置 |
| `Guidance` 取景提示 + 逐条约束结果 | `fast_loop.step` | 界面；拍摄时由 `session` 连同帧一起攒进片段 |
| `Take` 一条落盘的片段 | `takes.TakeStore.save` | `slow_loop.settle` |
| `JudgeRequest` → `JudgeVerdict` | `slow_loop` 提问 → `judge/` 回答 | 回到 `slow_loop` |
| `Plan` 镜头清单 | `plans.py`（手写）或 `plan_gen.PlanGenerator`（模型写、程序校验） | `planner` 把它包成 `PlanLoaded` 写进账本 |
| 事件（`ledger/events.py`） | `planner`：`PlanLoaded`；`slow_loop`：`TakeSettled`、`SceneFactObserved` | `Ledger.commit` → `reducer.apply` |
| `LedgerState`（`ledger/state.py`） | `Ledger.snapshot()`，或 `commit()` 的返回值 | `session`（算当前镜头、给界面）、`fast_loop.load`（快照）、`slow_loop` |
| `Settlement` 结算结果 | `slow_loop.settle` | `session`：通过就进下一个镜头，不通过原位重拍 |

### 谁可以 import 谁

每一行只 import 排在它上面的行（同一行里 `A <- B` 表示 B import A），没有反过来的，也没有环：

```
地基      contracts  config  texts  media  llm             不 import 任何内部模块
几何      spatial
账本      ledger/state  <-  events  <-  reducer  <-  store   读的一方只碰 state
零件      plans  perception/*  judge/*  takes  frame_source
谓词      constraints                                       用 spatial 和 ledger/state
生成器    plan_gen      用 constraints（只为 PREDICATES 这张表）
回路      fast_loop     用 constraints、perception/base、ledger/state（没有写入侧）
          slow_loop     用 constraints、judge/base、ledger/state + events + store
          planner       用 plans、plan_gen、ledger/events + store
状态机    session       用 fast_loop、slow_loop、planner、takes、ledger/state
装配      wiring  <-  server  <-  run.py
```

这张表是可执行的：`tests/test_architecture.py` 把"谁可以 import 谁"逐个模块写成了一张表，每次跑测试都对着真实代码检查。
"快回路不写账本"靠的是结构——它不 import 账本的写入侧，`wiring.py` 也不把 `Ledger` 对象交给它。

### 一帧的旅程（快回路）

1. `sources.js` 取一帧 → `POST /api/frame`（等回复回来才发下一帧，所以服务端永远只处理最新的画面）
2. `frame_source.ingest` 解码成 `Frame`
3. `session.on_frame`：取景或拍摄中 → 交给 `fast_loop.step`
4. `fast_loop.step`：检测器出 `Observation` → 逐条约束 `constraints.evaluate` → 挑最该说的一句
5. 拍摄中的话，`session` 把"这一帧 + 快回路对它的结果"攒进当前片段
6. 回复带回提示、检测框、按账本找回的东西；界面画出来

### 一条片段的旅程（慢回路）

1. 开拍：`session` 建一个 `TakeBuffer`，起一个定时器（固定 5 秒，可手动提前结束）
2. 结束：`session` 进入"检查中"，后台执行 `takes.save` → `slow_loop.settle`
3. `settle`：汇总逐帧的快约束结果 → 都过了才问 `judge` → 通过的话从逐帧检测里汇总场景事实 →
   一次 `ledger.commit([...], writer="slow_loop")`
4. `ledger/reducer.py` 套规则，`store.py` 落盘，版本号加一
5. `session` 回到取景：重新读账本、算出当前镜头、把新快照交给快回路

### 账本

- 文件：`data/ledger.json`（当前状态）+ `data/ledger.events.jsonl`（每次写入的完整记录，只追加）
- 两层：`shots`（镜头日志）、`scene`（场景状态）。每条约束的结论、每条场景事实都带来源：`plan` / `yolo` / `cosmos` / `sensor`
- 写入只有 `Ledger.commit()` 一个口，谁能提交什么写死在 `store.WRITE_PERMISSIONS`
- 规则全在 `ledger/reducer.py`：每个镜头一条有效素材（新的通过才顶掉旧的；都不过留最新的）；
  位置信 YOLO、语义信 Cosmos；场景事实只能来自已通过的有效素材，素材被替换时它带来的事实一并作废
- 被替换的片段文件留在 `data/takes/` 里，只是不在索引里了

## 文档没写到、我做了决定的地方

搭的过程中有些地方文档没定，或者两处说法合不上。下面是我的取舍，都可以改：

1. **快回路不写账本，可位置类字段又要来自 YOLO。** 做法：快回路把每帧看到的东西随提示交出去，拍摄时攒在片段里；
   慢回路结算时汇总，以 `source=yolo` 代写。（`slow_loop._position_fact`）
2. **一条片段怎样算通过。** 约束表里每条约束有自己的检查者，所以：快约束按"满足的帧占比 ≥ 60%"汇总，
   慢约束由 Cosmos 回答，两边都过才算过。快约束没过就不再问 Cosmos。（`slow_loop.settle`、`Tuning.pass_ratio`）
3. **多了一个 `session.py`。** 建议的模块里没有它，但"现在拍哪个镜头、这一帧给谁、什么时候结算"需要有人管。
   当前镜头不存，每次从账本现算。
4. **"拍摄开始时一次性加载快照"我理解成每个镜头开始取景时加载一次。** 整场只加载一次的话，镜头二看不到镜头一刚写下的事实，演示瞬间二就不成立。
5. **降级做成了一个统一的机制。** 能力 = 检测器的能力 ∪ 视频源的传感器；约束需要的能力不具备就退化成文字提示，并说明缺什么。
   没装 YOLO、没有姿态模型、没有陀螺仪走的是同一条路。降级的约束不参与通过判定，账本里记"没检测"，不冒充通过。
6. **看不见的东西按账本找回。** 包被挡住时，如果账本记着它相对桌子的位置、而桌子还看得见，快回路就按账本把它投影回当前画面（界面上是虚线框）。
   实时检测优先；两者左右相反时提示一句、以画面为准。
7. **距离只量水平方向。** 人坐着、包在桌下，高低差不代表离得远。（`spatial.gap_ratio`）
8. **3 个镜头是我选的。** S01 产出场景事实，S02 就是文档里那张约束表，S03 的俯角约束是为了把"传感器降级"这条路走通加的。
9. **每帧一个 HTTP 请求，没用 WebSocket。** 自带"只处理最新一帧"，每个接口能用 curl 单独试。
10. **快回路的 YOLO 在本地跑。** 主办方的 YOLO 端点收的是视频片段，不适合逐帧（见下一节）。
11. **语义约束多了一个英文问法 `ask`。** 给人看的描述是中文，给 Cosmos 的判断题用英文。
12. **加了"模拟画面"。** 文档里没有，是为了不依赖摄像头和模型也能验证回路。
13. **生成清单是一次调用加校验，不是多轮的 agent。** 文档 2.2 写的是"风格卡片 → 清单"，这里做成了"想法或脚本 → 清单"，
    没有风格卡片这一层。模型只填"拍什么、查什么"，ID、检查者、提示措辞由程序按谓词的规格填。
14. **模型能用的角色和谓词是从代码里现算的菜单。** 登记一个新谓词、在标签映射里加一种道具，下次生成就能用上，不用改提示词。
15. **只跟踪每种角色一个。** 所以提示词要求故事里最多一个人出镜；多人的戏现在写不了。
16. **快回路只保留清单里的角色。** 标签映射里加了椅子、杯子等十几种道具后，和当前清单无关的检测不往下传。

## 接真实模型

**YOLO**：`pip install ultralytics` 后默认就用（`yolo11n` + `yolo11n-pose`，首次运行自动下载权重到当前目录——
赛前在自己的网络下先跑一次，别等到现场再下）。
先跑 `python scripts/check_yolo.py`，把包、桌子、人摆成演示时的样子，看认得稳不稳；它会列出画面里所有被认出的类别，
包被认成别的类时在 `config.DEFAULT_LABEL_MAP` 加一行。

**Cosmos**：`COSMOS3_REASON_URL=... python run.py --judge cosmos`。变量名和主办方 VM 上的一致。
第一次接上先跑 `python scripts/probe_endpoints.py`，看模型回了什么、一条片段来回多久。
拿到真实端点之前可以用 `python scripts/fake_endpoints.py` 起一个同样接口形状的假端点练手。

### 主办方环境对照（来自入门仓库 relaxedtomato/vast-builders-challenge，2026-09-30 的版本）

| 设计文档里的问题 | 入门仓库怎么写的 | 对这份骨架意味着什么 |
|---|---|---|
| Cosmos 怎么调用 | `$COSMOS3_REASON_URL/v1/chat/completions`，OpenAI 兼容，模型 `nvidia/cosmos3-reason`，视频以 `data:video/mp4;base64,…` 内嵌，不需要鉴权 | `judge/cosmos.py` 就是照这个写的。注意：当天的指南里说这些模型"不直接调用、查它们生成的结果"，gpu 目录的说明里又给了直接调用的方法——能不能直接调、有没有限速，要现场确认 |
| YOLO 怎么调用 | `$YOLO_URL/v1/infer`，请求体是 `video_base64`（整段视频），自定义服务，有 `/openapi.json` | 逐帧的快回路用本地 YOLO；远程返回的结构用 `probe_endpoints.py` 存下来再决定用不用 |
| 向量 | `$COSMOS_EMBED1_URL/v1/embeddings`，256 维 | 以后做语义搜索时接在账本的描述字段上 |
| 在哪里开发 | 浏览器里的远程 VM，凭据和端点在 VM 的环境变量里；只能做 web 应用、CLI 或脚本 | 摄像头在你的笔记本上，服务跑在哪边要现场定，见下 |
| 能不能传自己的视频 | ingest 的说明写着这次只支持对已有素材 re-ingest，直接上传不在支持的流程里（仓库里又留着一个 upload-video 的 skill，两处不一致） | 账本自己就是索引，不依赖主办方的入库；想让自己拍的素材进他们的搜索，要现场问 |
| 自己逻辑用的 LLM | Weights & Biases 的 serverless inference，密钥在环境变量里 | 以后"风格卡片 → 清单"接它，改 `planner.make_plan` |

**服务跑在哪边**（这是我的推断，入门仓库没写）：浏览器只在 `https` 或 `localhost` 下开放摄像头。

- 服务跑在笔记本上（`localhost`，摄像头没问题）：需要 Cosmos 端点能从笔记本直接连上。到场第一件事就是在笔记本上跑 `probe_endpoints.py` 试。
- 连不上的话，服务得跑在 VM 上：笔记本的浏览器要用 `https` 地址访问它（需要隧道或主办方给的入口），帧走网络到 VM，快回路的延迟会多出一个来回。

## 还没做的、已知的问题

没验证过的：
- 生成清单这一步没连过真实的 OpenAI：请求格式、校验、重试对着假服务端测过，但 `gpt-5.6-luna` 实际写出来的清单质量、
  要等多久、它接不接受现在这种请求，都要你拿密钥跑 `scripts/try_plan.py` 才知道。模型名和价格是从第三方价格页查的，
  没在 OpenAI 官方页面核对上
- 主办方的 Weights & Biases 推理服务是不是 OpenAI 兼容接口、有哪些模型，没核实过
- 新加的那些道具（椅子、杯子、书……）YOLO 认得稳不稳没测过，只是 COCO 里有这些类别
- 真实的 Cosmos 端点没连过（请求格式和解析对着假服务端测过）
- 你的包和桌子 YOLO 认不认得出没测过，而且包是弱项。拿 COCO128（128 张带标注的样例图，还是模型自己的训练数据）量了一下：
  标了包的 14 张图里，`yolo11n` 只在 6 张里认出包，`yolo11s` 是 9 张；标了餐桌的 10 张，两个模型都认出 8 张。
  所以文档里"换颜色鲜明的道具"这条退路值得提前准备，也可以直接试 `--yolo-model yolo11s.pt`
- 人的朝向只对着照片调过：门槛是拿 COCO128 里三十来张有人的照片定的，正脸、侧脸、背身大体分得对；
  低头、半侧这类介于两者之间的会算成"面向镜头"。真人在摄像头前转头的效果没试过，用 `check_yolo.py` 看
- 真实摄像头没测过（浏览器这一侧用的是 Chromium 的假摄像头设备）
- 测试在 Linux 上用 Python 3.10 到 3.13 都跑过（其中 3.10 到 3.12 没装 ultralytics，那 4 个 YOLO 测试是跳过的）；
  浏览器只试过 Chromium。macOS / Windows、Safari / Firefox 没试过

骨架里有意没做的：
- "请确认"只有标记（置信度低的场景事实在界面上标出来），没有确认这个动作
- 同类检测到多个时取置信度最高的那个（多人、多个包的场景会选错）
- 生成的清单没有"先看一眼再决定用不用"这一步：生成完直接替换当前清单（会先问一句，因为要清空进度）
- 提示没有做时间上的平滑，检测抖动时提示会跟着跳
- 片段就是上送的抽帧，不是全帧率视频（`takes.py` 顶部写了怎么换成 MediaRecorder 补传）
- 账本里的左右是按画面方向记的，机位绕到桌子另一侧会反
- 重拍前面的镜头改了场景状态后，后面已经通过的镜头不会被标记为需要复查
- 界面和提示是中文的。要换英文：镜头相关的话在 `plans.py`，通用措辞在 `texts.py`，只在一处说的句子留在说它的模块里
  （清单见 `texts.py` 开头）；前端是 `web/index.html` 和 `web/js/` 下的几个文件

设计文档"暂缓项"的接入点：

| 暂缓项 | 接在哪 |
|---|---|
| 自动判断镜头起止 | `session.py`：在 `on_frame` 里根据 `guidance.ready` 调 `start_take` / `stop_take`，手动按钮保留 |
| 风格卡片 | 想法 / 脚本 → 清单已经做了（`plan_gen.py`）。风格卡片可以作为一段参考资料拼进 `plan_gen.system_prompt` |
| 让模型先问几个问题再定稿 | `session.replan` 现在是一问一答；要多轮就在 `plan_gen.PlanGenerator` 里加对话，输出还是一个 `Plan` |
| 语义搜索 | 读 `ledger.snapshot()` 里各片段的 `description`；向量用 Embed1 |
| 自动粗剪 | 按 `shots` 的顺序取各自 `effective_take.clip_dir`，ffmpeg 拼接 |
| 完整的冲突仲裁、修改留痕 | `ledger/reducer.py` 加事件和规则；写入记录已经是只追加的 |
| GO Ultra（RTMP） | 写一个拉流抽帧的进程，像浏览器一样调 `/api/source` 和 `/api/frame` |
| 遮挡物约束 | `constraints.py` 加一个谓词，登记到 `PREDICATES` |

## 目录

```
run.py                    启动入口
shotagent/                后端（从 __init__.py 的说明读起）
web/                      界面：index.html、style.css、js/{main,api,sources,render}.js
scripts/simulate.py       终端里讲解版的全流程
scripts/check_yolo.py     YOLO 认不认得出你的包、桌子、人
scripts/probe_endpoints.py  探主办方的 Cosmos / YOLO / Embed 端点
scripts/fake_endpoints.py   同样接口形状的假端点（也能冒充生成清单的模型）
scripts/try_plan.py       单独试一次：想法 / 脚本 → 镜头清单
tests/                    test_architecture（依赖规则）、test_ledger、test_spatial_constraints、
                          test_fast_loop、test_slow_loop、test_flow_http（整条回路）、
                          test_judge_cosmos、test_synthetic_media、test_yolo_optional、
                          test_plan_gen（生成器和模型客户端）、test_plan_flow（换清单的整条路）
data/                     运行时生成：ledger.json、ledger.events.jsonl、takes/
```
