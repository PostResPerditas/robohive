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

# Task1.2
base: /data/Project/robohive 中的内容通过 conda 环境 robohive 运行
base: 本机装有 cuda 但是你可能没有权限检测到；涉及训练工作请启用 cuda
目前我们的代码只是单个环境，没有吃满 gpu 的资源，能否对 scripts/remove.py 进行修改，让其支持并行训练，以增加训练效率。注意使用新的 py 文件

对 scripts/remove.py 和 scripts/train_parallel.py 添加功能，实现中途 checkpoint 和周期性 eval

python scripts/train_shadowhand_inhand_algos.py --list-tasks

python scripts/train_shadowhand_inhand_algos.py \
  --config scripts/config/pen/train_pen_ppo_parallel.json
python scripts/visualize_algos.py \
  --config scripts/config/pen/visualize_pen_ppo.json

# Task1.3
首先修改当前 robohive 项目的内容，基于 pen re orientation ppo 工作，写一个支持 tabletop 的重定位场景，尤其是手掌朝下，需要同时支持重力作用下的物体重定位工作。
注意你可能需要考虑 1. 有一个初始的抓取状态，使得物体首先被手抓取，而不是掉落；
2. 或者你可以让手被初始化在桌面，然后设置一个灵巧手的初始位姿，使得手可以从桌面捡起来物体，再作重定位，这样可以避免 1 中可能存在的掉落问题。注意我希望手的基坐标被固定，这样可以避免高纬度控制问题；当然，为了避免桌面场景对物体的干扰，可以手动指定基坐标上升的距离，将灵巧手基坐标的运动从 rl 中排除出去即可。

给定一批 refine/Dexonomy 生成的稳定抓取状态
从 init grasp reset
给定 target grasp goal
训练 ShadowHand 在重力下从 init 转到 target
成功条件 = object pose + required finger contact groups + no drop

# Task1.4
base: refine 项目使用 conda 环境 refine
base: robohive 项目使用 conda 环境 robohive
refine 中生成的抓取结果位于 grasp_refine/refine/outputs/grasp_generation_release/final/grasp_set；首先需要读取，或者仿真方式，剔除掉不稳定的抓取结果，测试结果可见 grasp_refine/refine/outputs/grasp_generation_release/evaluation/mujoco_stability
导入已有结果作为初始配置，注意物体应当被正确地初始化在桌面上
首先我们需要修改来实现重力作用下的抓取，尤其需要配置合适的初始抓取状态，使得手掌向下的抓取成为可能
已有工作表明，直接上真实重力场景可能会导致训练完全不可行，可以考虑通过重力课程逐渐增加重力强度，直至恢复真实值

整理 refine 中已通过 MuJoCo gravity hold 的 tabletop 抓取结果：

conda run -n refine python scripts/refine_grasp_dataset.py \
  --config scripts/config/refine/refine_grasp_dataset_tabletop.json

输出：

- runs/refine_grasps/tabletop_stable/manifest.json
- runs/refine_grasps/tabletop_stable/grasp_states.npz

其中 refine qpos 的格式为 hand xyz + hand quat wxyz + 22 个 ShadowHand 手指关节；脚本会额外导出 RoboHive/Adroit 顺序的 24 维 hand qpos，前两个 wrist 关节默认为 0。

训练脚本已支持重力课程，配置中可加入：

```json
{
  "gravity_vector": [0.0, 0.0, -9.81],
  "gravity_scale": 0.0,
  "eval_gravity_scale": 1.0,
  "gravity_curriculum": {
    "enabled": true,
    "start_scale": 0.0,
    "end_scale": 1.0,
    "duration_timesteps": 1000000,
    "start_timestep": 0
  }
}
```

使用 refine tabletop 稳定抓取状态作为 `relocate-v1` 初始状态，并逐步增加重力：

conda run -n robohive python scripts/train_parallel_algos.py \
  --config scripts/config/refine/train_refine_relocate_sac_gravity_curriculum.json

可视化训练出的模型：

conda run -n robohive python scripts/visualize_algos.py \
  --config scripts/config/refine/visualize_refine_relocate_sac.json

说明：当前 `RefineGraspResetWrapper` 是 RoboHive `relocate-v1` 的状态注入层，使用 refine 的稳定 grasp qpos 和 object pose，坐标默认采用 `[x, y, z]_refine -> [x, -z, y]_robohive`，将 refine tabletop 的 `+Y` 桌面法向对齐到 RoboHive 的 `+Z`。由于 `relocate-v1` 仍是内置 sphere 物体，而不是 refine 的真实 mesh，默认使用 `palm_object_offset_override` 把 RoboHive 的 `S_grasp` site 对齐到 sphere 附近，以保证初始手-物接触；真实物体 mesh 版本需要后续新增 MJCF/env。

python scripts/train_parallel_algos.py \
  --config scripts/config/refine/train_refine_relocate_sac_gravity_curriculum.json

python scripts/train_parallel_algos.py \
  --config scripts/config/refine/train_refine_relocate_ppo_gravity_curriculum.json

python scripts/visualize_algos.py \
  --config scripts/config/refine/visualize_refine_relocate_ppo.json

# Task1.5
当前的配置肯定还是不合适，需要引入正确的模型文件，否则抓取配置和物体无法统一起来；此外，我希望固定灵巧手的基坐标位姿，在完成初始化后，进行手内重定位到新的状态，不更改灵巧手的基坐标
模型文件的位置可以参考 /data/Project/Grasp_Refine/grasp_refine/assets/object/DGN_5k/processed_data，理论上在对应训练结果中应该也有注明

当前新增 `refine-tabletop-v1` 路线：不再使用 RoboHive `relocate-v1` 的 sphere 物体，而是根据 `runs/refine_grasps/tabletop_stable/manifest.json` 中记录的 `object_xml_path/object_scale/refine_hand_pose_world_wxyz/refine_finger_qpos22/object_scene_pose_wxyz` 生成固定基座 ShadowHand + 真实物体 mesh 的 MJCF。

关键点：

- 手基座固定为 refine 抓取结果中的 `refine_hand_pose_world_wxyz`，不进入 qpos/action。
- hand qpos 为 22 维 Refine ShadowHand 手指关节，object qpos 为 7 维 freejoint，所以 `nq=29`。
- action 为 18 维 ShadowHand actuator，使用 delta control，默认控制量由 joint/tendon transmission 从初始抓取 qpos 计算。
- 物体 XML 来自 `/data/Project/Grasp_Refine/grasp_refine/assets/object/DGN_5k/processed_data/.../urdf/coacd.xml`，路径由 manifest 中的 `object_xml_path` 读取。
- 默认碰撞过滤为 `object_hand_table`：手只和物体碰撞，物体和手/桌面碰撞，避免固定手基座场景下手-手自碰撞导致 reset 后数值发散。

训练固定基座、真实物体 mesh 的 tabletop reorientation PPO：

```bash
conda run -n robohive python scripts/train_parallel_algos.py \
  --config scripts/config/refine/train_refine_tabletop_ppo_fixed_base.json
```

可视化训练后的 best model：

```bash
conda run -n robohive python scripts/visualize_algos.py \
  --config scripts/config/refine/visualize_refine_tabletop_ppo_fixed_base.json
```

如果只想检查环境初始化和一步仿真，可以临时用 Python 直接构建 `RefineTabletopEnv`；期望现象是 `nq=29, nu=18`，reset 后有物体-手接触，零动作在 `gravity_scale=1.0` 下短时稳定。

# Command

讨论：我希望最终实现桌面抓取 -> 重定位到目标状态的工作；但是我目前存在几个歧路点：1. 目前的重定位操作是手掌向上，这对桌面抓取的情况不符合，因为抓取和重定位过程存在重力约束；2. 目前的 robohive 重定位目标仅包含物体位姿，而不包含包含手指接触在内的约束，例如可以不要求目标状态的所有接触点被重复，但是每根手指代表的接触组应当至少要保持在目标状态附近；3. 考虑到重定位过程中需要保持物体的稳定，我们是否可以考虑依赖优化或者学习策略，对整个重定位过程进行中间阶段生成，用于引导 rl 过程抵达最终状态；4. 如果依赖 3，整体论文的科学性是否会受到干扰；5. 如果依赖上述内容，我们需要预先输入目标状态和初始状态 [如果需要输入初始状态的话]，这样不可避免地会引起随机性的减弱，我担心会影响学习过程，此外，还需要基于先前的 refine 项目生成足够多的抓取候选，这对数量和质量是否存在较高要求。当前存在的问题和可做工作太多，我一时无法想好首先从哪一个点开始，哪一个点最重要。
