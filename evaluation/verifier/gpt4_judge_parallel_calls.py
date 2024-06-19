import os
import time
import json
from tqdm import tqdm
import concurrent.futures
import argparse
from gpt4_judge import Judge

from text_dataset_files_constants import POISONED_TEST_DATASET_FILENAME, MODELS_RESPONSE_OUT_FILENAME_PER_MODEL, VERIFIER_RESPONSE_OUT_FILENAME_PER_MODEL

MODEL = 'llama3_8b' 
MAX_THREADS = 60 #Change this if failure rate is too high 
JUDGE_PROMPT_FILE = 'judge_prompt.txt' #Prompt to give to the judge 
JUDGE_MODEL = 'gpt-4-no-filter' ##name of Azure model deployment 
START_IDX = 0 
END_IDX = -1
AZURE_OPENAI_KEY = '' #add cred.
AZURE_OPENAI_ENDPOINT = '' 

RESPONSE_OUT_FILENAME = MODELS_RESPONSE_OUT_FILENAME_PER_MODEL[MODEL] ##generated responses file 
VERIFIER_OUTPUT_FILENAME =  VERIFIER_RESPONSE_OUT_FILENAME_PER_MODEL[MODEL] ##output file


os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_KEY
os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT


def find_not_finished_items(all_data, verifier_output):
    not_finished = []
    for key_ in range(0,len(all_data)):
        if not str(key_) in verifier_output:
            not_finished.append(key_)
            continue
        if len(verifier_output[str(key_)]) == 0:
            not_finished.append(key_)
            continue
        if verifier_output[str(key_)]['short'] == '':
            not_finished.append(key_)
            continue           
    return not_finished 

class Counter:
    """Monitors the status of our threads. Used for debugging and monitoring.
    """
    running = 0
    waiting = 0
    failed = 0
    def __init__(self, pbar):
        self.pbar = pbar
        self.display()

    def display(self):
        self.pbar.set_description(f"Running: {self.running}, Waiting: {self.waiting}, Failed: {self.failed}. Progress")

    def update(self, waiting=0, running=0, failed=0):
        self.waiting += waiting
        self.running += running
        self.failed += failed
        self.display()


verifier_output = {}
failed_ids = []

def foo_wrapper(i, data_item, response_item):
    """Contains all the thread logic for launching the function, including sleeping and error handling.

    If it fails, it waits 2x the time it waited before, until a limit of >32
    (i.e., when ~1 minute has passed).
    """
    print(i)
    global counter
    if not "counter" in globals():
        counter = Counter(tqdm(disable=True))
    counter.update(running=1)

    wait_seconds = 1
    while wait_seconds <= 32:
        try:
            res = judge.process_item(data_item,response_item)
            counter.update(running=-1)
            return res
        except Exception as e:
             # Fetch error code.
            msg = e.args[0]
            if msg.startswith("Error code: ") and msg.split(" ")[2] == "429":
                # Rate limit exceeded. We'll wait.
                pass
            else:
                counter.update(failed=1, running=-1)
                print(f"Unhandled error: {e}")

        # Wait.
        counter.update(waiting=1)
        time.sleep(wait_seconds)
        wait_seconds *= 2
        counter.update(waiting=-1)

    # We were killed by the rate limit. Won't try anymore.
    print("Failed after trying for more than 1 minute.")
    counter.update(failed=1, running=-1)
    return []

def launch(all_data_items, all_response_items):
    global counter # Gives us fancy stats and a progress bar.

    # appending to file from futures: https://stackoverflow.com/questions/57621086/concurrency-in-read-and-write-json-file-by-multiple-thread-in-python
    with tqdm(total=len(all_data_items)) as pbar:
        for i in all_data_items.keys():
            if i in verifier_output:
                pbar.update(1)

        # Global counter for threads stats data.
        counter = Counter(pbar)

        with open(VERIFIER_OUTPUT_FILENAME, "a") as f:
            # We can use a with statement to ensure threads are cleaned up promptly
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
                # Start the load operations and mark each future with its doc_id.
                
                future_to_doc = {
                    executor.submit(
                        foo_wrapper,
                        i = doc_id,
                        data_item=all_data_items[doc_id],
                        response_item= all_response_items[doc_id]
                    ): doc_id for doc_id in all_data_items.keys()
                    }
                for future in concurrent.futures.as_completed(future_to_doc):
                    doc_id = future_to_doc[future]
                    try:
                        results = future.result()
                    except Exception as exc:
                        print(f"Failed to get responses {doc_id}: {exc}")
                        failed_ids.append(doc_id)
                    else:
                        verifier_output[doc_id] = results
                        results = {doc_id: results}
                        f.write(json.dumps(results) + "\n")
                        f.flush()
                    pbar.update(1)




data = json.load(open(POISONED_TEST_DATASET_FILENAME))
responses = json.load(open(RESPONSE_OUT_FILENAME))
assert len(data) == len(responses)

judge = Judge(JUDGE_PROMPT_FILE, JUDGE_MODEL)
print(f'Start index: {START_IDX}')
END_IDX = len(responses) if END_IDX == -1 else END_IDX 
print(f'End index: {END_IDX}')

data_subset = {str(i): data[i] for i in range(START_IDX,END_IDX)}
responses_subset = {str(i): responses[str(i)] for i in range(START_IDX,END_IDX)}

launch(data_subset, responses_subset) 


not_finished = find_not_finished_items(data_subset, verifier_output)

## Retry for examples that failed 
while True:
    print(f'=== Running for the remaining {len(not_finished)} items === ')
    new_data_subset = {str(i): data_subset[i] for i in not_finished}
    new_responses_subset = {str(i): responses_subset[str(i)] for i in not_finished}
    launch(new_data_subset, new_responses_subset) 
    not_finished = find_not_finished_items(data_subset, verifier_output)
    if len(not_finished) == 0: break 