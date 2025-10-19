import gradio as gr
from sentence_transformers import SentenceTransformer
from utils import search_on_db, embed_text, parse_db_response

model = SentenceTransformer("all-MiniLM-L6-v2")


def make_search(text: str) -> str:
    embed = embed_text(text, model)
    res = search_on_db(embed)
    results = parse_db_response(res)

    cards_html = ""
    for r in results:
        cards_html += f"""
        <div style="display:inline-block; width:250px; margin:10px; border:1px solid #ddd; border-radius:10px; overflow:hidden; box-shadow:0 2px 6px rgba(0,0,0,0.1);">
            <img src="{r["img"]}" style="width:100%; height:150px; object-fit:cover;">
            <div style="padding:10px;">
                <h4 style="margin:0; font-size:16px;">{r["name"]}</h4>
            </div>
        </div>
        """

    return cards_html


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Sidebar():
        gr.Markdown("# Motor de búsqueda semántica")
        text_area = gr.TextArea(
            label="Búsqueda", placeholder="Introduce el texto de búsqueda"
        )
        button = gr.Button(value="Consultar")

    gr.Markdown("# Resultados de la búsqueda")
    output = gr.HTML()

    button.click(fn=make_search, inputs=text_area, outputs=output)


demo.launch(share=False)
