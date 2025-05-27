# Send Queries to the NVIDIA Triton Server for NeMo LLMs

After starting the service with the scripts supplied in the TensorRT-LLM, vLLM, and In-Framework sections, the service will be in standby mode, ready to receive incoming requests. There are multiple methods available for sending queries to this service.

* Use the Query Script: Execute the query script within the currently running container.
* PyTriton: Utilize PyTriton to send requests directly.
* HTTP Requests: Make HTTP requests using various tools or libraries.


## Send a Query using the Script

The following example shows how to execute the query script within the currently running container.

1. To use a query script, run the following command:

   ```shell
   python scripts/deploy/nlp/query.py --url "http://localhost:8000" --model_name nemotron --prompt "What is the capital of United States?"
   ```
   
2. Change the url and the ``model_name`` based on your server and the model name of your service. The code in the script can be used as a basis for your client code as well.

3. If the there is a prompt embedding table, run the following command to send a query:

   ```shell
   python scripts/deploy/nlp/query.py --url "http://localhost:8000" --model_name nemotron --prompt "What is the capital of United States?" --task_id "task 1"
   ```
   
4. The following parameters are defined in the ``deploy_triton.py`` script:

   - ``--url``: url for the triton server. Default="0.0.0.0".
   - ``--model_name``: name of the triton model to query.
   - ``--prompt``: user prompt.
   - ``--max_output_len``: Max output token length. Default=128.
   - ``--top_k``: considers only the top N most likely tokens at each step.
   - ``--top_p``: determines the cumulative probability distribution used for sampling the next token in the generated response. Controls the diversity of the output.
   - ``--temperature``: controls the randomness of the generated output. Higher value, such as 1.0, leads to more randomness and diversity in the generated text, a lower value, like 0.2, produces more focused and deterministic responses.
   - ``--task_id``: id of a task if ptuning is enabled.
   

## Send a Query using the NeMo APIs

The NeMo Framework provides NemoQueryLLM APIs to send a query to the Triton server for convenience. These APIs are only accessible from the NeMo Framework container.

1. To run the request example using NeMo APIs, run the following command:

   ```python
   from nemo.deploy.nlp import NemoQueryLLM

   nq = NemoQueryLLM(url="localhost:8000", model_name="nemotron")
   output = nq.query_llm(prompts=["What is the capital of United States?"], max_output_len=10, top_k=1, top_p=0.0, temperature=1.0)
   print(output)
   ```

2. Change the url and the ``model_name`` based on your server and the model name of your service. Please check the NeMoQuery docstrings for details.

3. If there is a prompt embedding table, run the following command to send a query:

   ```python
   output = nq.query_llm(prompts=["What is the capital of United States?"], max_output_len=10, top_k=1, top_p=0.0, temperature=1.0, task_id="0")
   ```