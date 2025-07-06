# Internals Analysis

## Smaller extraction
```
python h_internals/src/f1_extract.py \
  --input_json a_data/alpaca/merge_instructs/all.json \
  --out_file act_attn_200.pt \
  --hf_key <HF_TOK> \
  --paraphrase_types instruct_all_caps instruct_american_english \
  instruct_apologetic instruct_apology instruct_archaic \
  instruct_australian_english instruct_authorative instruct_causal_chat \
  instruct_child_directed instruct_chinese_simplified instruct_colloquial \
  instruct_command instruct_condensed_then_expand \
  instruct_condensed_then_expand_with_examples \
  instruct_condensed_then_expand_with_examples_and_explanations \
  instruct_csv_line instruct_double_negative instruct_email \
  instruct_emergency_alert instruct_emoji instruct_emoji_only \
  instruct_esperanto instruct_fewest_words \
  instruct_formal_business instruct_french instruct_gamer_slang \
  instruct_gaming_jargon instruct_german instruct_html_tags \
  instruct_inline_url instruct_insulting \
  instruct_joke instruct_leet_speak instruct_lyrical instruct_markdown_italic \
  instruct_meta_question instruct_morse_code instruct_no_caps \
  instruct_no_spaces instruct_poetic \
  instruct_positive instruct_random_caps \
  instruct_random_linebreaks instruct_rap_verse instruct_salesy \
  instruct_sarcastic instruct_sceptical \
  instruct_second_person instruct_silly instruct_spanglish instruct_spanish \
  instruct_therapy_session instruct_tweet instruct_typo_extra_letter \
  instruct_typo_missing_vowel \
  instruct_typo_random instruct_typo_repeated_letter instruct_urgent \
  instruct_with_additional_context instruct_with_summary instruct_witty \
  instruct_yaml_block instruction_original \
  --n_samples 200 \
  --model google/gemma-2b-it \
  --layers auto \
  --log_tag act_attn_200
```

## Larger extraction
```
python h_internals/src/f1_extract.py \
  --input_json a_data/alpaca/merge_instructs/all.json \
  --out_file act_attn_500.pt \
  --hf_key <HF_TOK> \
  --paraphrase_types instruct_absurdist instruct_advertisement \
  instruct_ambiguous_scope \
  instruct_all_caps instruct_american_english \
  instruct_apologetic instruct_apology instruct_archaic \
  instruct_australian_english instruct_authorative instruct_causal_chat \
  instruct_casual_chat \
  instruct_child_directed instruct_chinese_simplified instruct_colloquial \
  instruct_command instruct_condensed_then_expand \
  instruct_condensed_then_expand_with_examples \
  instruct_condensed_then_expand_with_examples_and_explanations \
  instruct_contradictory_ask \
  instruct_csv_line instruct_debate_turns instruct_double_negative \
  instruct_dual_audience \
  instruct_email instruct_emergency_alert instruct_emoji \
  instruct_emoji_and_typo instruct_emoji_and_typo_and_random_question_marks \
  instruct_emoji_only \
  instruct_empty_input instruct_esperanto instruct_few_words \
  instruct_fewest_words \
  instruct_first_plural instruct_first_singular instruct_formal_academic \
  instruct_formal_business instruct_forum_quote instruct_french \
  instruct_gamer_slang \
  instruct_gaming_jargon instruct_german instruct_haiku instruct_hopeful \
  instruct_html_tags \
  instruct_inline_url instruct_insulting \
  instruct_joke instruct_leet_speak instruct_lyrical instruct_markdown_italic \
  instruct_melancholy instruct_meta_question instruct_might_be_wrong \
  instruct_modal_may instruct_modal_must instruct_modal_should \
  instruct_morse_code \
  instruct_nested_parentheticals instruct_nested_question instruct_no_caps \
  instruct_news_headline \
  instruct_no_spaces instruct_poetic \
  instruct_positive instruct_random_caps \
  instruct_random_linebreaks instruct_rap_verse instruct_salesy \
  instruct_sarcastic instruct_sceptical \
  instruct_second_person instruct_silly instruct_spanglish instruct_spanish \
  instruct_therapy_session instruct_tweet instruct_typo_extra_letter \
  instruct_typo_missing_vowel \
  instruct_typo_random instruct_typo_repeated_letter instruct_urgent \
  instruct_with_stream_of_consciousness instruct_with_detailed_instructions \
  instruct_with_additional_context instruct_with_summary instruct_witty \
  instruct_yaml_block instruction_original \
  --n_samples 500 \
  --model google/gemma-2b-it \
  --layers auto \
  --log_tag act_attn_500
```

## start attn analysis

logdir="logs"
mkdir -p "$logdir"
ts() { date -u +"%Y-%m-%dT%H-%M-%SZ"; }
python h_internals/src/h01_metrics.py \
  --extract_file ../paraphrx_internals/act_attn_200.pt \
  --scores_file c_assess_inf/output/alpaca_answer_scores/gemma-2-2b-it.json \
  --out_prefix runs/gemma200 \
  >>"$logdir/h01_metrics_$(ts).log" 2>&1 &
disown
