import torch
from pathlib import Path
import os
from mellow import MellowWrapper
    
if __name__ == "__main__":
    # setup cuda and device
    cuda = torch.cuda.is_available()
    device = 0 if cuda else "cpu"

    # setup mellow
    mellow = MellowWrapper(
                        config="conf.yaml",
                        model = "v0.ckpt",
                        device=device,
                        use_cuda=cuda,
                    )
    
    # pick up audio file paths
    parent_path = Path(os.path.realpath(__file__)).parent
    path1 = os.path.join(parent_path, "resource", "1.wav")
    path2 = os.path.join(parent_path, "resource", "2.wav")
    
    # list of filepaths and prompts
    examples = [
        [path1, path2, "Based on the audio, what can be said about the hypothesis - \"A farmer is giving a tour of his ranch while chickens roam nearby\"? a) It is definitely true b) It is definitely false c) It is plausible d) I cannot determine"],
    ]

    # generate response
    response = mellow.generate(examples=examples, max_len=300, top_p=0.8, temperature=1.0)
    print(f"\noutput: {response}")