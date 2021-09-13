from models.semantic_captioning_character_lstm import SemanticCaptioningCharacterLSTM

daily_dialog_generator = SemanticCaptioningCharacterLSTM(recursion_depth=100,path_to_model_weights="models/lstm_dailydialog.hdf5")
while True:
    conversation = input("\n\n> user: ")
    print(f"> LSTM-DailyDialog (top-p=.75, temperature=1.5): {daily_dialog_generator.generate(conversation,top_p=.75,temperature=1.5)}")
    print(f"> LSTM-DailyDialog (top-p=.75, temperature=1.2): {daily_dialog_generator.generate(conversation,top_p=.75,temperature=1.2)}")
    print(f"> LSTM-DailyDialog (top-p=.75, temperature=.9): {daily_dialog_generator.generate(conversation,top_p=.75,temperature=.9)}")
    print(f"> LSTM-DailyDialog (top-p=.75, temperature=.7): {daily_dialog_generator.generate(conversation,top_p=.75,temperature=.7)}")
    print(f"> LSTM-DailyDialog (top-p=.75, temperature=.5): {daily_dialog_generator.generate(conversation,top_p=.75,temperature=.5)}")

    print(f"> LSTM-DailyDialog (top-p=.95, temperature=1.5): {daily_dialog_generator.generate(conversation,top_p=.95,temperature=1.5)}")
    print(f"> LSTM-DailyDialog (top-p=.95, temperature=1.2): {daily_dialog_generator.generate(conversation,top_p=.95,temperature=1.2)}")
    print(f"> LSTM-DailyDialog (top-p=.95, temperature=.9): {daily_dialog_generator.generate(conversation,top_p=.95,temperature=.9)}")
    print(f"> LSTM-DailyDialog (top-p=.95, temperature=.7): {daily_dialog_generator.generate(conversation,top_p=.95,temperature=.7)}")
    print(f"> LSTM-DailyDialog (top-p=.95, temperature=.5): {daily_dialog_generator.generate(conversation,top_p=.95,temperature=.5)}")
