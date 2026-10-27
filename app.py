import streamlit as st
import pymupdf as fitz
from PIL import Image
import io
import torch
import base64

st.set_page_config(page_title="Multimodal RAG (BLIP + BERT + Generator)", layout="wide")

st.title("📚 Multimodal RAG System (BLIP + BERT + AI Generator)")
st.write("Extracts PDF content, captions diagrams with **BLIP**, embeds with **BERT**, and generates elaborated AI responses in Exam Mode!")

# --- SIDEBAR ---
st.sidebar.header("Document & Mode Setup")
uploaded_file = st.sidebar.file_uploader("Upload PDF Document", type=["pdf"])

rag_mode = st.sidebar.radio(
    "Select Retrieval Mode:",
    ["Grounded (Fact-Based)", "Exam-Driven (AI Elaborated Summary)"]
)

index_button = st.sidebar.button("Process & Index PDF")

# --- LAZY LOAD MODELS ---
@st.cache_resource
def load_all_models():
    from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, pipeline
    import chromadb

    # BLIP for images
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # BERT for embeddings
    bert_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    bert_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Generator for Exam Mode (Flan-T5 for local text generation)
    gen_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    generator = pipeline("text2text-generation", model=gen_model, tokenizer=gen_tokenizer, max_length=512)

    chroma_client = chromadb.Client()

    return blip_processor, blip_model, bert_tokenizer, bert_model, generator, chroma_client

def get_bert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    return embeddings

# --- PROCESSING ---
if uploaded_file is not None:
    if index_button:
        with st.spinner("Initializing models (BLIP, BERT, Generator) and indexing document..."):
            blip_proc, blip_mod, bert_tok, bert_mod, generator, client = load_all_models()
            
            mode_key = "grounded" if "Grounded" in rag_mode else "exam_driven"
            collection = client.get_or_create_collection(name=f"rag_{mode_key}")
            
            bytes_data = uploaded_file.read()
            doc = fitz.open(stream=bytes_data, filetype="pdf")
            total_items = 0
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract page images
                images = page.get_images(full=True)
                page_img_b64 = ""
                if images:
                    base_image = doc.extract_image(images[0][0])
                    img_bytes = base_image["image"]
                    page_img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                
                # Text Stream
                text = page.get_text().strip()
                if text:
                    text_emb = get_bert_embedding(text, bert_tok, bert_mod)
                    collection.add(
                        documents=[text],
                        embeddings=[text_emb],
                        metadatas=[{"source": "text", "page": page_num + 1, "image_base64": page_img_b64}],
                        ids=[f"p{page_num}_txt"]
                    )
                    total_items += 1
                
                # Visual Stream (BLIP)
                for idx, img in enumerate(images):
                    base_image = doc.extract_image(img[0])
                    img_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                    
                    inputs = blip_proc(image, return_tensors="pt")
                    caption = blip_proc.decode(blip_mod.generate(**inputs)[0], skip_special_tokens=True)
                    
                    cap_emb = get_bert_embedding(caption, bert_tok, bert_mod)
                    collection.add(
                        documents=[caption],
                        embeddings=[cap_emb],
                        metadatas=[{"source": "visual", "page": page_num + 1, "image_base64": img_b64}],
                        ids=[f"p{page_num}_img_{idx}"]
                    )
                    total_items += 1

            st.session_state["indexed"] = True
            st.session_state["collection_name"] = f"rag_{mode_key}"
            st.success(f"Successfully indexed {total_items} items with AI generation ready!")

# --- QUERY SECTION ---
if st.session_state.get("indexed"):
    st.divider()
    st.header("Query Your Document")
    query = st.text_input("Ask a question (e.g., 'Explain vapor compression cycle for 10 marks'):")
    
    if st.button("Search & Answer") and query:
        blip_proc, blip_mod, bert_tok, bert_mod, generator, client = load_all_models()
        collection = client.get_collection(name=st.session_state["collection_name"])
        
        query_emb = get_bert_embedding(query, bert_tok, bert_mod)
        results = collection.query(query_embeddings=[query_emb], n_results=3)
        
        q_lower = query.lower()
        is_10_marks = any(m in q_lower for m in ["10 mark", "ten mark", "10-mark", "essay", "detailed"])
        is_5_marks = any(m in q_lower for m in ["5 mark", "five mark", "5-mark", "short note", "brief"])
        
        st.subheader("Retrieval Results:")
        if results["documents"] and len(results["documents"][0]) > 0:
            is_grounded = "grounded" in st.session_state["collection_name"]
            shown_pages = set()
            
            for i, doc_text in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i]
                p_num = metadata["page"]
                source_type = metadata.get("source", "text")
                img_b64 = metadata.get("image_base64", "")
                
                if is_grounded:
                    st.info(f"### 📌 Grounded Fact Match (Page {p_num} - Source: {source_type.upper()})")
                    st.markdown(f"> *\"{doc_text}\"*")
                else:
                    # Exam-Driven Mode: Use AI Generator to elaborate!
                    st.warning(f"### 🎓 AI Exam-Driven Synthesis (Page {p_num})")
                    
                    if is_10_marks:
                        prompt = f"Provide a detailed, comprehensive 10-mark essay answer explaining this topic thoroughly: {query}. Base it on this context: {doc_text}"
                    elif is_5_marks:
                        prompt = f"Provide a concise 5-mark short note explaining: {query}. Base it on this context: {doc_text}"
                    else:
                        prompt = f"Explain this concept clearly for an exam study guide: {query}. Base it on this context: {doc_text}"
                    
                    with st.spinner("AI generating elaborated answer..."):
                        ai_response = generator(prompt, max_length=350, do_sample=False)[0]['generated_text']
                    
                    st.markdown(f"**🤖 AI Elaborated Answer:**\n\n {ai_response}")
                    st.markdown(f"❓ **Suggested Exam Question:** *Explain the working and significance of {query} with proper diagrams.*")
                
                # Render diagram if available
                if img_b64 and p_num not in shown_pages:
                    img_bytes = base64.b64decode(img_b64)
                    st.image(io.BytesIO(img_bytes), caption=f"🖼️ Reference Diagram Retrieved from Page {p_num}", width=500)
                    shown_pages.add(p_num)
                    
                st.divider()
        else:
            st.warning("No relevant information found.")
else:
    st.info("👈 Open the sidebar, upload your PDF, and click 'Process & Index PDF'.")