import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from gemma.tokenizer import Tokenizer
import sentencepiece as spm
tokenizer = Tokenizer('/home/medi/LLM/gemma_pytorch/tokenizer/tokenizer.model')
lines = [
  "`DEVOCEAN`은 SK그룹의 대표 개발자 커뮤니티이자🧑",
  "내/외부 개발자 간 소통과 성장을 위한 플랫폼을 상징합니다.👋",
  "`Developers`' Ocean 개발자들을 위한 영감의 바다🙏",
  "`Devotion` 헌신,몰두,전념💯",
  "`Technology for Everyone` 모두를 위한 기술👍"
  ]

for line in lines:
    inputs = tokenizer.encode(line)    
    print(inputs)  
    decoded_sequence = tokenizer.decode(inputs[0])
    print(decoded_sequence)
    print()
    print(tokenizer.decode(7))
