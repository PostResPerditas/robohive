# Task1
base: /data/Project/robohive 中的内容通过 conda 环境 robohive 运行
base: 本机装有 cuda 但是你可能没有权限检测到；涉及训练工作请启用 cuda
前情提要：我们尝试实现抓取状态生成和手内重定位工作，以实现 桌面抓取 -> 手内操作 -> 指定抓取状态的工作
先前的项目中，我们基于 dexonomy 项目完成了抓取状态生成，现在我希望通过 rl 方法实现手内重定位操作，以从初始接触状态转移到目标接触状态

我尝试通过 robohive 实现上述任务。我观察到有一些 manipulation 的内容，我想要知道如何调用当前已有的手内重定位相关的任务。给出示例和说明。

python -m robohive.utils.examine_env \
  -e relocate-v1 \
  -r onscreen \
  -n 1

# Task1.1
如果首先不考虑嵌入我的生成状态，只是完成 robohive 本身自带的抓取和移动任务，我应该使用什么命令。当前的
python -m robohive.utils.examine_env \
  -e relocate-v1 \
  -r onscreen \
  -n 1
命令显示灵巧手是随机运动，没有很好地完成任务。我如何给定训练任务，保存模型文件，读取模型文件并进行检验

讨论：当前 stable baseline3 的官方要求是 3.10+，老版本例如 1.8.0 虽然支持 py 3.8，但是其内部调用存在冲突

# Command
