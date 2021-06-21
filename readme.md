## Install
1. Recommand using Anaconda for version management

    conda create --name coverage python=3.6 \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    pip install gym==0.10.5 tensorflow==1.8.0 numpy==1.14.5
2. Install maddpg and multiagent-particle-envs to PythonPATH

    cd ~/Coverage/maddpg\
    pip install -e .\
    cd ~/Coverage/multiagent-particle-envs\
    pip install -e .

## Usage
    cd ~/Coverage/maddpg/experiments/
    python run_groups.py --scenario simple_spread --display --load-dir weight4v4/
