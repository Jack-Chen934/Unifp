# Go2-W 官方模型

本目录保留宇树官方 ROS 仓库中的 Go2-W 描述包，未修改上游 URDF、网格、关节配置或启动文件。

- 上游仓库：[unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros)
- 上游路径：`robots/go2w_description/`
- 固定提交：`3ae893d905d11795ddf1075825ee5010364a1109`（2026-10-30）
- 模型入口：`robots/go2w_description/urdf/go2w_description.urdf`
- 许可：根目录 `LICENSE`，BSD 3-Clause

上游 URDF 引用 `package://go2w_description/dae/` 下的网格，因此保留了原始 ROS 包目录名和仓库相对路径。该包包含 URDF、网格、关节名配置、RViz/Gazebo 启动文件及 `package.xml`。

下载时将各文件的 Git Blob 对象哈希与固定提交逐一核对。该目录是官方 ROS 描述模型；它尚未转换或导入 Isaac Gym/Isaac Lab，也不代表其中惯性、碰撞和执行器参数已经过 Go2-W 实机标定。
