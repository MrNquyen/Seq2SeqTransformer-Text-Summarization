python main.py \
--config ./config/config.yaml \
--save_dir ./save \
--run_type train \
--device cuda:3 
# --resume_file /datastore/npl/ViInfographicCaps/workspace/baseline_summarization/summarization_custom_vocab/Seq2SeqTransformer-Text-Summarization/save/checkpoints/model_1000.pth

# Scheduler within epochs
python main_modify.py \
--config ./config/config.yaml \
--save_dir ./save \
--run_type train \
--device cuda:4 \
--resume_file /data2/npl/ViInfographicCaps/workspace/baseline_summarization/summarization_custom_vocab/Seq2SeqTransformer-Text-Summarization/save/checkpoints/model_2240.pth
