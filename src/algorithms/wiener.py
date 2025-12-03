"""
Implementação do algoritmo Wiener para deconvolução.
"""

import numpy as np
from numpy.fft import fft2, ifft2, ifftshift
from .base import DeconvolutionAlgorithm


class Wiener(DeconvolutionAlgorithm):
    """
    Algoritmo Wiener para deconvolução de imagens.
    
    Um método baseado em filtragem no domínio da frequência que minimiza o erro quadrático médio.
    """
    
    @property
    def name(self):
        return "wiener"
    
    @property
    def description(self):
        return "Algoritmo Wiener - Método rápido no domínio da frequência"
    
    def deconvolve(self, image, psf, balance=0.01, clip=True, logger=None, **kwargs):
        """
        Aplica o algoritmo Wiener para deconvolução de imagem.

        O algoritmo Wiener é baseado na filtragem no domínio da frequência e resolve:
        G(u,v) = F(u,v) * [ H*(u,v) / (|H(u,v)|^2 + K) ]

        Onde:
        - G(u,v) é a transformada da imagem deconvoluída
        - F(u,v) é a transformada da imagem observada (borrada)
        - H(u,v) é a transformada da PSF (Point Spread Function)
        - H*(u,v) é o conjugado complexo de H(u,v)
        - K é o parâmetro de equilíbrio (balance)
        
        Args:
            image: Imagem de entrada (numpy.ndarray, pode ser RGB ou grayscale)
            psf: Point Spread Function (numpy.ndarray)
            balance: Parâmetro de equilíbrio K (float, padrão: 0.01)
            clip: Se True, limita os valores entre 0 e 1 após deconvolução (bool, padrão: True)
            logger: Logger opcional para mensagens de progresso (DeconvolutionLogger)
            **kwargs: Parâmetros adicionais (ignorados)
        
        Returns:
            numpy.ndarray: Imagem deconvoluída
        """
        # Verificar se a imagem é RGB ou grayscale
        is_rgb = len(image.shape) == 3 and image.shape[2] == 3
        
        # Converter balance para float caso venha como string
        try:
            balance = float(balance)
        except (ValueError, TypeError):
            balance = 0.01

        if logger:
            image_type = "RGB" if is_rgb else "Grayscale"
            logger.info(f"Iniciando deconvolução Wiener ({image_type}, balance={balance})")
        
        if is_rgb:
            # Processar cada canal separadamente
            deconvolved_channels = []
            for channel in range(3):
                channel_data = image[:, :, channel]
                deconvolved_channel = self._wiener_single_channel(
                    channel_data,
                    psf,
                    balance
                )
                deconvolved_channels.append(deconvolved_channel)
            
            deconvolved = np.stack(deconvolved_channels, axis=2)
        else:
            # Processar imagem em escala de cinza
            deconvolved = self._wiener_single_channel(
                image,
                psf,
                balance
            )
        
        # Aplicar clipping se solicitado
        if clip:
            deconvolved = np.clip(deconvolved, 0, 1)
            
        if logger:
            logger.info("Deconvolução Wiener concluída")
        
        return deconvolved
    
    def _wiener_single_channel(self, image, psf, balance):
        """
        Aplica o algoritmo Wiener em um único canal.
        
        Args:
            image: Imagem de entrada (numpy.ndarray 2D)
            psf: Point Spread Function (numpy.ndarray 2D)
            balance: Parâmetro de equilíbrio K
        
        Returns:
            numpy.ndarray: Imagem deconvoluída
        """
        # Dimensões
        h, w = image.shape
        ph, pw = psf.shape
        
        # 1. Preparar a PSF para a FFT (padding)
        psf_padded = np.zeros((h, w))
        
        # Centralizar a PSF na matriz de padding
        start_h = (h - ph) // 2
        start_w = (w - pw) // 2
        psf_padded[start_h:start_h+ph, start_w:start_w+pw] = psf
        
        # Mover o centro da PSF para a origem (0,0) para a FFT
        psf_padded = ifftshift(psf_padded)
        
        # 2. Transformar para o Domínio da Frequência (FFT)
        img_fft = fft2(image)
        psf_fft = fft2(psf_padded)
        
        # 3. Aplicar a Fórmula de Wiener
        # G(u,v) = F(u,v) * [ H*(u,v) / (|H(u,v)|^2 + K) ]
        
        psf_conj = np.conj(psf_fft)
        denominator = (np.abs(psf_fft) ** 2) + balance
        
        # Evitar divisão por zero
        denominator = np.maximum(denominator, 1e-10)
        
        # Calcular o resultado na frequência
        result_fft = img_fft * (psf_conj / denominator)
        
        # 4. Voltar para o Domínio Espacial (IFFT)
        result = np.real(ifft2(result_fft))
        
        return result