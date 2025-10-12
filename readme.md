这个仓库其实非常的乱，从原作者开始就非常的乱，经过我改完以后就更乱；暂时凑合着看。更多细节建议还是对照作者原仓库看，如果有兴趣的话还是建议修整一下仓库代码让看起来更舒服。
以下为我修改后的仓库几个重要指令的操作教程：

1. ppo算法训练  
需要注意这里分为初始观察和增强观察。初始观察基本沿用原作者送入ppo网络的状态向量，增强观察是我修改以后的。需要修改不同的配置代码才能进行不同的训练。  

```bash  
python openmines\src\utils\rl_data_collector\dqn_collector.py --env_config openmines\src\conf\north_pit_mine.json --episodes 50 --max_steps 2000 --env_id mine/Mine-v1 --use_enhanced_observation # 收集满足维度的正则化参数（如474维的）
# 放置到根目录下命名为normalization_params_474.json
python openmines/test/cleanrl/ppo_single_net.py --use-enhanced-observation
```

```bash  
查看训练曲线
tensorboard --logdir /home/chengrongxian/git/MineDisaptcher-openmines/runs --bind_all
tensorboard --logdir runs/ --port 6006
```

2. 实验
```bash
pip install -e . 安装仓库
openmines run -f C:\Users\95718\Desktop\vscode\MineDisaptcher-openmines\openmines\src\conf\north_pit_mine.json

```

3. 上传到仓库：
原作者仓库：https://github.com/370025263/openmines  
本人仓库：  
```bash
git status          # 检查，尤其检查.gitignore有没有包含大文件
git add .           # 添加
git commit -m "更新描述"  # 提交
git pull origin master    # 同步
git push origin master    # 推送
```

4. 服务器切换
我写了一套自动化流程部署脚本，可以自动传到服务器上代码，然后进行训练
但暂时还没有实践，也懒得实践了暂时