from models.data_loader_utils import load_daily_dialog
contexts,questions = zip(*load_daily_dialog("data/daily_dialog"))

from models.semantic_captioning_character_lstm import SemanticCaptioningCharacterLSTM
rnn_model = SemanticCaptioningCharacterLSTM(recursion_depth=100)
rnn_model.train(question_contexts=contexts,questions=questions,batch_size=6,epochs=100,save_to_file_path="lstm_dailydialog.hdf5") 