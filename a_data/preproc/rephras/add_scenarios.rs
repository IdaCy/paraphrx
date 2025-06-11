/* 
cargo add_scenarios < \
    a_data/mmlu/prxed_moral_500/voice.json > \
    a_data/mmlu/prxed_moral_500_scenarios/voice.json
*/

use std::io::{self, Read};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Slurp the entire stdin into a string
    let mut raw = String::new();
    io::stdin().read_to_string(&mut raw)?;

    // Parse as generic JSON
    let mut data: serde_json::Value = serde_json::from_str(&raw)?;

    // Expect a top-level JSON array
    let arr = data
        .as_array_mut()
        .ok_or("Input JSON must be an array of objects")?;

    for obj in arr {
        // Only proceed if this element is a JSON object
        let map = obj
            .as_object_mut()
            .ok_or("Array element is not a JSON object")?;

        // Pull out the original instruction string
        if let Some(instruction) = map
            .get("instruction_original")
            .and_then(|v| v.as_str())
        {
            if let Some(start) = instruction.find("Scenario 1") {
                let scenarios = &instruction[start..];
                // Insert (or overwrite) the new "scenarios" field
                map.insert(
                    "scenarios".into(),
                    serde_json::Value::String(scenarios.to_string()),
                );
            }
        }
    }

    // Pretty-print the patched JSON array
    println!("{}", serde_json::to_string_pretty(&data)?);
    Ok(())
}
