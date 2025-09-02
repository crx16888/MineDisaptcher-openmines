本部分是使用RL来进行dispatcher训练的脚本。

# 1.数据采集
为了实现状态和reward的正则化，防止训练过早进入次优解，因而要采集一些数据进行统计分析。
```shell
python3 openmines/src/utils/rl_data_collector/dqn_collector.py 
```
每个config中的策略会采集多个episode然后把所有算法放在一起计算state_mean、state_std,reward_mean,reward_std.
然后脚本会把结果保存为normalization_params.json 需要把他们改名放置在算法文件夹下，数据采集、训练、实验用同一个正则化文件。

# 2.模型训练
```shell
python3 openmines/test/cleanrl/ppo_single_net.py
```
训练脚本有很多参数可以选择。
核心的点在于：
    - 是否启用教师引导策略
    - 环境的奖励函数是密集的还是稀疏的
    - 是否采用了正则化（对训练来说极为重要‼️）
Note: 没有使用reward正则化RL训练会比不过经典策略。
## 2.1 openmines矿山训练环境
密集奖励环境:
    mine/Mine-v1-dense; mine/Mine-v1
    密集环境中会包含经过精心设计的奖励函数
稀疏奖励环境:
    mine/Mine-v1-sparse
    稀疏奖励仅仅会在最终输出总的tons产出(经过rescale)

# 3.模型导出和评测
你需要修改openmines/src/dispatch_algorithms/ppo_dispatcher.py中的model_path参数来让其生效。
model可以从checkpoints中进行寻找，通常训练的时候会将表现好的模型保存。

```shell
 CUDA_VISIBLE_DEVICES=5 openmines run -f /home/chengrongxian/git/openmines/openmines/src/conf/north_pit_mine.json
 # 使用特定显卡运行实验
```

