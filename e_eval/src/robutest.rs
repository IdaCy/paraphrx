
for vote_num in 1..=args.num_judge_votes {
    if api_calls_used >= args.api_call_max {
        logger.log("API call limit reached -> aborting early");
        break 'outer;
    }

    let mut success = false;
    let mut last_err: Option<anyhow::Error> = None;

    for attempt in 1..=3u8 {
        let api_key = match key_pool.take_key_for_call() {
            Ok(k) => k,
            Err(_) => {
                logger.log("All API keys exhausted -> aborting early");
                break 'outer;
            }
        };

        match query_gemini_for_judgment(
            &client,
            &api_key,
            &args.judging_model,
            &prompt_for_judge,
            args.judge_temperature,
        ).await {
            Ok((local_verdict, conf12)) => {
                *votes.entry(local_verdict).or_insert(0) += 1;
                vote_confidences.push(conf12.clamp(0.0, 1.0));
                success = true;
                break;
            }
            Err(e) if attempt < 3 => {
                last_err = Some(e);
                logger.log(&format!(
                    "ID {id} key {key}: vote {vote_num} attempt {attempt} failed; retrying..."
                ));
                sleep(Duration::from_millis(300 * (attempt as u64))).await;
            }
            Err(e) => {
                last_err = Some(e);
                logger.log(&format!(
                    "ID {id} key {key}: vote {vote_num} failed after retries."
                ));
                break;
            }
        }
    }

    if !success {
        if let Some(e) = last_err {
            logger.log(&format!("Final error: {e}"));
        }
        logger.log(&format!("ID {id} key {key}: skipping this vote due to failure"));
    }

    api_calls_used += 1;
    if args.delay_ms > 0 && args.num_judge_votes > 1 {
        sleep(Duration::from_millis(args.delay_ms)).await;
    }
}
