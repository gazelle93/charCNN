import torch
import torch.nn as nn
import string

from text_processing import word_tokenization


class CharCNN(nn.Module):
    def __init__(self, num_chars, num_channels, filters):
        super(CharCNN, self).__init__()
        self.embeddings = nn.Embedding(num_chars, num_channels)
        self.conv_layers = nn.ModuleList([nn.Conv1d(in_channels=num_channels,
                                     out_channels=num_f,
                                     kernel_size=width,
                                     bias=True) for width, num_f in filters])
        self.activation = nn.ReLU()

    def forward(self, inputs):
        convs = []
        # character embedding: num_tokens x num_characters
        character_embedding = torch.tensor(inputs, dtype=torch.long)

        # embedding matrix: num_tokens x num_characters x num_channels
        _inputs = self.embeddings(character_embedding)

        for conv_layer in self.conv_layers:
            convolved = conv_layer(torch.transpose(_inputs, 1, 2))
            convolved, _ = torch.max(convolved, dim=-1)
            convolved = self.activation(convolved)
            convs.append(convolved)

        # concatenated character embedding: num_tokens x projection_dim
        return torch.cat(convs, dim=-1)



class Highway(nn.Module):
    def __init__(self, input_dim, n_highway):
        super(Highway, self).__init__()
        self.highway_layers = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, input_dim * 2) for _ in range(n_highway)]
        )

        for h_layer in self.highway_layers:
            h_layer.bias[input_dim:].data.fill_(1)

        self.activation = nn.ReLU()

    def forward(self, inputs):
        current_input = inputs
        for h_layer in self.highway_layers:
            projected_input = h_layer(current_input)
            linear_part = current_input

            nonlinear_part, gate = projected_input.chunk(2, dim=-1)
            nonlinear_part = self.activation(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part

        return current_input



class Projection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Projection, self).__init__()
        self.projection_layer = torch.nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, inputs):
        return self.projection_layer(inputs)



def get_chars():
    temp_chars = [x for x in string.printable]
    temp_chars += ["<BOW>", "<EOW>", "<PAD>"]

    char_dict = {}
    reverse_char_dict = {}
    for idx, ch in enumerate(temp_chars):
        char_dict[ch] = idx
        reverse_char_dict[idx] = ch

    return char_dict, reverse_char_dict

def get_padded_input(inputs, char_dict, max_characters_per_token):
    input_list = []
    for tk in inputs:
        temp_idx_list = [char_dict["<PAD>"] for x in range(max_characters_per_token)]
        temp_idx_list[0] = char_dict["<BOW>"]

        for idx, ch in enumerate(tk[:max_characters_per_token-2]):
            temp_idx_list[idx+1] = char_dict[ch]
            last_idx = idx+2

        temp_idx_list[last_idx] = char_dict["<EOW>"]

        input_list.append(temp_idx_list)
    return input_list


def get_chardict_and_padded_input(inputs, max_characters_per_token):
    char_dict, reverse_char_dict = get_chars()

    padded_input = get_padded_input(inputs, char_dict, max_characters_per_token)
    return char_dict, padded_input


def get_models(num_chars, num_channels, filters, n_highway, output_dim):
    input_dim = sum([x[1] for x in filters])
    CharCNN_l = CharCNN(num_chars, num_channels, filters)
    Highway_l = Highway(input_dim, n_highway)
    Projection_l = Projection(input_dim, output_dim)
    return CharCNN_l, Highway_l, Projection_l


def get_word_embeddings(CharCNN_l, Highway_l, Projection_l, input):
    token_emb = CharCNN_l(input)
    token_emb = Highway_l(token_emb)
    token_emb = Projection_l(token_emb)

    return token_emb
