# LLM + RAG Setup 📚 

This is a simple template for generating UI elements in React using LLMs. After a good amount
of experimentation, the `Mistral-7b-Instruct-v0.1` model has been chosen for both tokenization 
and as the main LLM model. Given its relatively small size and better accuracy to size ratio, 
it seemed perfect for the use case. Although it runs locally on a MacBook Pro (M2, 16GB), 
the quantization is not properly configured, leading to slower results and occasional garbage values.

For a `RAG` (Retrieval-Augmented Generation) setup, embeddings need to be created first. 
The `FAISS` library has been used for this purpose. The task is to generate UI components in React. 
While the base model is proficient at generating JSX and HTML, it struggles with styling. 
This is where RAG becomes essential. Before feeding data to the LLM, a large store of embeddings 
is created from websites like `Bootstrap`, `Tailwind`, etc. Various tools from the LangChain library, 
such as `AsyncChromiumLoader`, `Html2TextTransformer` have been utilized for this purpose.

### Setup ⚙

The code presented here is a result of curiosity-driven experimentation, so the workflow may not be optimal.

> [!NOTE]
> RAG creation is done locally, while LLM prompting is performed in Google Colab.
> <br> RAG creation involves merging multiple embeddings into a one.

**To create embeddings and RAG:**

1. Gather all required links for RAG creation
2. Place these links in a text file under the `data` folder
3. Use the `embeddings.py` script to generate vector embeddings for that article
4. Use the `create_rag` function in `data_loader.py` to create a RAG

> [!TIP]
> Check out the `rag_mistral_7b.py` file to get a good idea on how to create and use RAGs
> <br> We will be using a somewhat modified version of it in Colab though

**To prompt the LLM:**

1. Open the `RAG_Mistral_7b.ipynb` notebook in Google Colab
2. Upload your embeddings (in the notebook, we call it `bootstrap_rag`)
3. Run the notebook cell by cell to understand it better
