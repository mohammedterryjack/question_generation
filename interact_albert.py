from models.albert_masktoken import AlbertMaskToken

albert_generator = AlbertMaskToken()
while True:
    conversation = input("\n\n> user: ")
    print(f"> Baseline: {albert_generator.quick_generate(conversation)}")
    print(f"> Albert: {albert_generator.generate(conversation)}")