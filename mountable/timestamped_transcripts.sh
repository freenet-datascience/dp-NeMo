nemo_model="/workspace/NeMo/mountable/dev_model/stt_de_conformer_ctc_large.nemo"
language_model="/workspace/NeMo/mountable/dev_model/result_topchoice_frienett.bin"
probs_path="/workspace/NeMo/mountable/dev_model/probs.pickle"
vocab_path="/workspace/NeMo/mountable/dev_model/vocab.pickle"
tokenizer_path="/workspace/NeMo/mountable/dev_model/tokenizer.pickle" 
manifest_path="/opt/data/bi_adm/personal/mhalbe/manifest_0510_mini.json"
sounds_like_csv="/workspace/NeMo/mountable/my_converter/custom_words_nemo.csv"
preds_path="/workspace/NeMo/mountable/dev_model/outScript0510mini/"
parlance_path="/workspace/NeMo/mountable/dev_model/outScript0510mini/parlance/"
result_path="/workspace/NeMo/mountable/dev_model/outScript0510mini/combined/"

cp /workspace/NeMo/mountable/eval_beamsearch_ngram_plus.py /workspace/NeMo/scripts/asr_language_modeling/ngram_lm/eval_beamsearch_ngram_plus.py

python3 /workspace/NeMo/scripts/asr_language_modeling/ngram_lm/eval_beamsearch_ngram_plus.py --nemo_model_file $nemo_model --kenlm_model_file $language_model --probs_cache_file $probs_path --vocab_cache_file $vocab_path --tokenizer_cache_file $tokenizer_path --input_manifest $manifest_path --beam_width 64 --beam_alpha 1.0 --beam_beta 0.5 --preds_output_folder $preds_path --device cpu --beam_batch_size 4 --acoustic_batch_size 4 --sounds_like_csv $sounds_like_csv

python3 /workspace/NeMo/mountable/parlance_decoder.py -t $tokenizer_path -v $vocab_path -p $probs_path -l $language_model -o $parlance_path

for f in $parlance_path
do
	python3 /workspace/NeMo/mountable/my_converter/soundsLikeReplacer.py $f result
done

python3 /workspace/NeMo/mountable/my_converter/nemoMatcher.py --good ${preds_path}/top_choice.tsv --folderJson $parlance_path --output $result_path