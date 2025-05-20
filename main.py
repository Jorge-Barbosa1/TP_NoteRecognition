import cv2
from inference import InferencePipeline

# Função para processar cada frame
def my_sink(result, video_frame):
    if result.get("output_image"):
        # Mostrar imagem com bounding boxes
        cv2.imshow("Acordes - Roboflow", result["output_image"].numpy_image)
        
        # Mostrar o nome do acorde mais provável
        predictions = result.get("predictions", [])
        if predictions:
            top_pred = max(predictions, key=lambda x: x['confidence'])
            print(f"Acorde: {top_pred['class']} ({top_pred['confidence']*100:.1f}%)")
        else:
            print("Nenhum acorde detetado.")
        
        # Fechar ao carregar em 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pipeline.stop()

# Inicializar pipeline com as tuas credenciais
pipeline = InferencePipeline.init_with_workflow(
    api_key="YtazWlskmOkf0xesjewF", 
    workspace_name="aopmod2",
    workflow_id="detect-count-and-visualize",
    video_reference=0,  # 0 para webcam local
    max_fps=30,
    on_prediction=my_sink
)

pipeline.start()
pipeline.join()
