from models.data_loader_utils import load_quora_data
train_data = load_quora_data('data/quora_duplicate_questions.tsv')

from models.semantic_captioning_character_lstm import SemanticCaptioningCharacterLSTM
rnn_model = SemanticCaptioningCharacterLSTM(
    recursion_depth=100,
    path_to_model_weights="models/partially_trained_quora.hdf5"
)
rnn_model.train(train_data,batch_size=6,epochs=500,save_to_file_path="lstm_quora.hdf5") 