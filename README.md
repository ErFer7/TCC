# TCC - Fine-tuning do MLLM LLaMA para a classificação de lesões de pele

Lesões de pele podem ser um indicativo de diversas doenças, incluindo doenças graves
como o câncer de pele. A detecção precoce dessas lesões é fundamental para o tratamento
e cura da doença. Porém, o diagnóstico e classificação de uma lesão de pele é normalmente
feita por profissionais especializados em hospitais ou clínicas. Isto pode levar a um atraso
no diagnóstico pela falta de acesso ou procura pelo atendimento médico.

Considerando este cenário, tecnologias como Multimodal _Large Language Models_ (MLLMs)
podem ser úteis. Estes modelos podem identificar lesões de pele com base em imagens e
prover um pré-diagnóstico que pode alertar o portador da lesão sobre a necessidade de
procurar atendimento médico. O modelo _Large Language and Vision Assistant_ (LLaVa)
é um bom candidato para esta aplicação, pois consegue descrever imagens e pode ser
adaptado para propósitos específicos.

Neste trabalho, propõe-se a adaptação do LLaVa com técnicas de _fine tuning_ para classificar lesões de pele com uma precisão aceitável.
