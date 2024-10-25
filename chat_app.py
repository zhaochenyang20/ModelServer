import gradio as gr
from model_server import ModelServer

CSS = """
.gradio-container { height: 100vh !important; }
#chatbot-container { display: flex; flex-direction: column; height: 100%; }
#chatbot-panel { flex-grow: 1; overflow: auto; height: auto !important; }
"""

def create_chat_message(role: str, content: str) -> dict:
    return {"role": role, "content": content}

class ChatBot:
    def __init__(self):
        self.model_server = ModelServer()
        self.chat_history = []
        
    def chat(self, message: str, history: list, temperature: float, max_tokens: int, model_size: str) -> str:
        self.chat_history.append(create_chat_message("user", message))
        
        try:
            response = self.model_server.get_completion_or_embedding(
                model_size=model_size,
                message=self.chat_history,
                temperature=temperature,
                max_tokens=max_tokens
            )
            self.chat_history.append(create_chat_message("assistant", response))
            return response
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_embedding(self, text: str, k: int) -> str:
        try:
            embedding = self.model_server.get_completion_or_embedding(
                model_size="7",
                message=text,
                get_embedding=True
            )
            # 确保 k 不超过 embedding 的长度
            k = min(k, len(embedding))
            # 返回前 k 个元素，格式化为易读的字符串
            return f"First {k} elements of embedding:\n" + "\n".join([f"{i}: {val}" for i, val in enumerate(embedding[:k])])
        except Exception as e:
            return f"Error: {str(e)}"

def main():
    chatbot = ChatBot()
    
    with gr.Blocks(css=CSS) as demo:
        with gr.Tabs():
            with gr.Tab("Chat"):
                gr.Markdown("# AI Assistant")
                gr.Markdown("Chat with AI using the ModelServer backend.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        temperature = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature",
                            info="Controls randomness (0 = deterministic, 1 = creative)"
                        )
                        max_tokens = gr.Slider(
                            minimum=64,
                            maximum=2048,
                            value=512,
                            step=64,
                            label="Max Tokens",
                            info="Maximum length of the response"
                        )
                        model_size = gr.Radio(
                            choices=["8", "70"],
                            value="8",
                            label="Model Size",
                            info="Choose between 8B and 70B models"
                        )
                    
                    with gr.Column(scale=3, elem_id="chatbot-container"):
                        chatbot_interface = gr.ChatInterface(
                            fn=lambda message, history, temp, tokens, model: chatbot.chat(
                                message, history, temp, tokens, model
                            ),
                            additional_inputs=[temperature, max_tokens, model_size],
                        )
                
                gr.Examples(
                    examples=[
                        ["Hello, how are you?"],
                        ["What can you help me with?"],
                        ["Tell me a short story."]
                    ],
                    inputs=chatbot_interface.textbox,
                )
            
            with gr.Tab("Embedding"):
                gr.Markdown("# Get Embeddings")
                gr.Markdown("Get embeddings for your text input.")
                
                with gr.Row():
                    with gr.Column():
                        text_input = gr.Textbox(
                            label="Input Text",
                            placeholder="Enter text to get embedding...",
                            lines=5
                        )
                        k_value = gr.Number(
                            label="Number of elements to show",
                            value=10,
                            minimum=1,
                            step=1
                        )
                        embed_btn = gr.Button("Get Embedding")
                    
                    with gr.Column():
                        output = gr.Textbox(
                            label="Embedding Result",
                            lines=10,
                        )
                
                embed_btn.click(
                    fn=chatbot.get_embedding,
                    inputs=[text_input, k_value],
                    outputs=output
                )

    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()