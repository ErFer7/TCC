# TCC - Fine-tuning do MLLM LLaMA para a Classificação de Lesões de Pele

Lesões de pele podem ser um indicativo de diversas doenças, incluindo doenças graves como o câncer de pele. A detecção precoce dessas lesões é fundamental para o tratamento e cura da doença. Porém, o diagnóstico pode ser feito somente por profissionais qualificados, como dermatologistas.

Uma parte do atendimento de atenção primária no Brasil é feita por Agentes Comunitários de Saúde (ACS). Esses profissionais estão em contato direto com a população, porém, eles não são qualificados para realizar a triagem de casos de lesões de pele. Considerando este cenário, uma ferramenta capaz de classificar lesões de pele e fornecer pré-diagnósticos pode ser útil.

Multimodal Large Language Models (MLLMs) possuem as capacidades necessárias para serem utilizados no desenvolvimento de uma ferramenta como esta. Estes modelos podem identificar e classificar lesões de pele com base em imagens e prover um pré-diagnóstico compreensível, indicando a gravidade do problema e a urgência da busca pelo atendimento médico. Além disso, MLLMs podem ser adaptados para tarefas especializadas através de fine-tuning.

Neste trabalho, propõe-se a adaptação do MLLM Large Language Model Meta AI (LLaMA) 3.2 com diferentes técnicas de fine-tuning baseadas em Parameter-Efficient Fine-Tuning (PEFT), como o Quantized Low Rank Adaptation (QLoRA) e Low-Rank Adaptation (LoRA), comparando-as entre si, para classificar lesões de pele com uma acurácia aceitável.