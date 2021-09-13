from models.data_loader_utils import load_squad_data
contexts,questions = zip(*load_squad_data("data/train-v2.0.json"))

from models.semantic_captioning_character_lstm import SemanticCaptioningCharacterLSTM
rnn_model = SemanticCaptioningCharacterLSTM(recursion_depth=100,path_to_model_weights="models/partially_trained_squad2.hdf5")
rnn_model.train(question_contexts=contexts,questions=questions,batch_size=6,epochs=100,save_to_file_path="lstm_squad2.hdf5") 