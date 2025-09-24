1. 实验
pip install -e . 安装仓库
openmines run -f ...

1.1 查看训练曲线
tensorboard --logdir /home/chengrongxian/git/MineDisaptcher-openmines/runs --bind_all


2. 上传到仓库：
git add .
git commit -m "你的提交说明"
git push origin master


3. 推理切换194维度和384维度：
class PPODispatcher(BaseDispatcher):
    def __init__(self, model_path: Optional[str] = None, use_enhanced_observation: bool = False):

4. 训练切换194维度和384维度：
在训练之前注意先收集正则化参数，运行utils\rl_data_collector\dqn_collector.py

# 在 ppo_single_net.py 的 Args 类中
@dataclass 
class Args:
    # ...其他参数
    use_enhanced_observation: bool = True  # ✅ 默认使用384维

# 使用384维训练
python openmines/test/cleanrl/ppo_single_net.py --use_enhanced_observation=True

# 使用194维训练  
python openmines/test/cleanrl/ppo_single_net.py --use_enhanced_observation=False

5. 服务器切换
我们做了一套自动化流程部署脚本，可以自动传到服务器上代码，然后进行训练
但还没有实践