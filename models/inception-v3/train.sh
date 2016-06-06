python train.py \
--data_dir=/data/tom/iLID-Data/output/2016-05-31_spectrograms_color_299x299/tf_records \
--train_dir=`pwd`/snapshots \
--max_steps=50000 \
--subset='train' \
--num_gpus=1 \
--fine_tune=True \
--pretrained_model_checkpoint_path=`pwd`/model.ckpt-157585 \
--initial_learning_rate=0.001 \
--num_epochs_per_decay=300.0 \
--learning_rate_decay_factor=0.16 \
--input_queue_memory_factor=8 \
--batch_size=32