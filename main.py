import argparse
from text_processing import get_nlp_pipeline, word_tokenization
import elmoCharacterEncoder

def main(args):
    cur_text = "People create a computational process."
    selected_nlp_pipeline = get_nlp_pipeline(args.nlp_pipeline)

    # Initialize Character Encoding of ELMo
    # Filters for CharCNN part
    filters =  [[1, 4], [2, 8], [3, 16], [4, 32], [5, 64]]

    tokens = word_tokenization(cur_text, selected_nlp_pipeline, args.nlp_pipeline)

    char_dict, padded_input = elmoCharacterEncoder.get_chardict_and_padded_input(tokens, args.max_characters_per_token)

    CharCNN_l, Highway_l, Projection_l = elmoCharacterEncoder.get_models(len(char_dict), args.num_channels, filters, args.n_highway, args.output_dim)

    initial_embeddings = elmoCharacterEncoder.get_word_embeddings(CharCNN_l, Highway_l, Projection_l, padded_input)


    # Feed intialized embedding to BiLSTM
    print("Result of CharCNN embeddings.")
    for tk, emb in zip(tokens, initial_embeddings):
        print("{}: {}".format(tk, emb))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--nlp_pipeline", default="spacy", type=str, help="NLP preprocessing pipeline.")
    parser.add_argument("--num_channels", default=10, type=int, help="The size of word embedding.")
    parser.add_argument("--output_dim", default=128, type=int, help="The size of output dimension.")
    parser.add_argument("--max_characters_per_token", default=50, type=int, help="The maximum size of characters of token.")
    parser.add_argument("--n_highway", default=2, type=int, help="The number of highway layers.")
    args = parser.parse_args()

    main(args)
