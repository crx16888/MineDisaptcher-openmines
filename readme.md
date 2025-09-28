这个仓库其实非常的乱，从原作者开始就非常的乱，经过我改完以后就更乱；暂时凑合着看。更多细节建议还是对照作者原仓库看。
以下为我修改后的仓库几个重要指令的操作教程：

1. ppo算法训练  
需要注意这里分为初始观察和增强观察。初始观察基本沿用原作者送入ppo网络的状态向量，增强观察是我修改以后的。需要修改不同的配置代码才能进行不同的训练。  

在训练之前注意先收集正则化参数，运行utils\rl_data_collector\dqn_collector.py（但我已经在ppo_single_net.py中集成，直接运行即可）

如使用增强观察训练：
```bash  
python openmines/test/cleanrl/ppo_single_net.py --use_enhanced_observation=True
```

```bash  
查看训练曲线
tensorboard --logdir /home/chengrongxian/git/MineDisaptcher-openmines/runs --bind_all
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
git add .
git commit -m "你的提交说明"
git push origin master
```

4. 服务器切换
我写了一套自动化流程部署脚本，可以自动传到服务器上代码，然后进行训练
但暂时还没有实践，也懒得实践了暂时