import yaml
from omegaconf import OmegaConf
from modules.sac_module import SACModule

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning import loggers as pl_loggers
import os
import datetime

from src.build import build_env

def main():
    base_save_dir = './result/train'
    
    # 実行時のタイムスタンプを付与して、一意のディレクトリ名を生成
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(base_save_dir, timestamp)

    config_paths = [
        './config/config.yaml',
    ]

    # 各 YAML ファイルを読み込んで OmegaConf にマージ
    configs = [OmegaConf.load(path) for path in config_paths]
    merged_conf = OmegaConf.merge(*configs)
    
    # 統合されたconfigを保存
    os.makedirs(save_dir, exist_ok=True)  # 保存ディレクトリを作成
    merged_config_path = os.path.join(save_dir, 'merged_config.yaml')
    with open(merged_config_path, 'w') as f:
        yaml.dump(OmegaConf.to_container(merged_conf, resolve=True), f)
    
    env = build_env(config=merged_conf)
    # データモジュールとモデルモジュールのインスタンスを作成
    model = SACModule(env=env, config=merged_conf)

    checkpoint_callback = ModelCheckpoint(
        monitor='actor_loss',
        mode='min',
        filename='sac-{epoch:02d}-{actor_loss:.2f}',
        save_top_k=3,
        verbose=True,
    )

    

    callbacks = [checkpoint_callback]


    # TensorBoard Loggerもsave_dirに対応させる
    logger = pl_loggers.TensorBoardLogger(
        save_dir=base_save_dir,  # ベースディレクトリ（ここでlightning_logsが作成される）
        name='',  # デフォルトの`lightning_logs`ディレクトリに保存
        version=timestamp  # タイムスタンプをバージョン名に使用
    )

    # トレーナーを設定
    trainer = pl.Trainer(
        max_epochs=merged_conf.max_epochs,
        max_steps=merged_conf.max_steps,
        logger=logger,  # Loggerに対応させる
        callbacks=callbacks,
        accelerator='gpu',
        devices=[0],  # 使用するGPUのIDのリスト
        benchmark=True,  # cudnn.benchmarkを使用して高速化
    )

    # モデルの学習を実行
    trainer.fit(model)

if __name__ == '__main__':
    main()
