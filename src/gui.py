"""
Interface gráfica para deconvolução de imagens.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import numpy as np
from PIL import Image, ImageTk
import threading
import os
from .psf_generator import generate_gaussian_psf, generate_motion_psf
from .deconvolution import deconvolve, get_available_algorithms
from .logger import DeconvolutionLogger


class DeconvolutionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Deconvolução de Imagens")
        self.root.geometry("1400x700")
        
        self.image_path = None
        self.original_image = None
        self.deconvolved_image = None
        
        # Obter algoritmos disponíveis
        self.available_algorithms = get_available_algorithms()
        
        self.setup_ui()
    
    def setup_ui(self):
        # Frame superior para controles
        control_frame = tk.Frame(self.root, padx=10, pady=10)
        control_frame.pack(fill=tk.X)
        
        # Botão para selecionar imagem
        select_btn = tk.Button(
            control_frame,
            text="Selecionar Imagem",
            command=self.select_image,
            font=("Arial", 10),
            padx=10,
            pady=5
        )
        select_btn.pack(side=tk.LEFT, padx=5)
        
        # Label para mostrar caminho da imagem
        self.image_label = tk.Label(
            control_frame,
            text="Nenhuma imagem selecionada",
            font=("Arial", 9),
            fg="gray"
        )
        self.image_label.pack(side=tk.LEFT, padx=10)
        
        # Separador
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Frame para parâmetros do algoritmo
        params_frame = tk.Frame(control_frame)
        params_frame.pack(side=tk.LEFT, padx=10)
        
        # Label e dropdown para algoritmo de deconvolução
        tk.Label(params_frame, text="Algoritmo:", font=("Arial", 9)).grid(row=0, column=0, padx=5, sticky=tk.W)
        self.algorithm_var = tk.StringVar(value=self.available_algorithms[0] if self.available_algorithms else "richardson_lucy")
        algorithm_combo = ttk.Combobox(
            params_frame,
            textvariable=self.algorithm_var,
            values=self.available_algorithms,
            state="readonly",
            width=18,
            font=("Arial", 9)
        )
        algorithm_combo.grid(row=0, column=1, padx=5)
        
        # Label e dropdown para tipo de blur
        tk.Label(params_frame, text="Tipo de Blur:", font=("Arial", 9)).grid(row=1, column=0, padx=5, sticky=tk.W)
        self.blur_type_var = tk.StringVar(value="gaussian")
        blur_type_combo = ttk.Combobox(
            params_frame,
            textvariable=self.blur_type_var,
            values=["gaussian", "motion"],
            state="readonly",
            width=12,
            font=("Arial", 9)
        )
        blur_type_combo.grid(row=1, column=1, padx=5)
        blur_type_combo.bind("<<ComboboxSelected>>", self.on_blur_type_change)
        
        # Frame para parâmetros gaussianos
        self.gaussian_frame = tk.Frame(params_frame)
        self.gaussian_frame.grid(row=2, column=0, columnspan=2, pady=5, sticky=tk.W)
        
        tk.Label(self.gaussian_frame, text="Tamanho:", font=("Arial", 9)).grid(row=0, column=0, padx=5, sticky=tk.W)
        self.size_var = tk.StringVar(value="15")
        size_entry = tk.Entry(self.gaussian_frame, textvariable=self.size_var, width=8, font=("Arial", 9))
        size_entry.grid(row=0, column=1, padx=5)
        
        tk.Label(self.gaussian_frame, text="Sigma:", font=("Arial", 9)).grid(row=0, column=2, padx=5, sticky=tk.W)
        self.sigma_var = tk.StringVar(value="5.0")
        sigma_entry = tk.Entry(self.gaussian_frame, textvariable=self.sigma_var, width=8, font=("Arial", 9))
        sigma_entry.grid(row=0, column=3, padx=5)
        
        # Frame para parâmetros de movimento
        self.motion_frame = tk.Frame(params_frame)
        self.motion_frame.grid(row=2, column=0, columnspan=2, pady=5, sticky=tk.W)
        self.motion_frame.grid_remove()  # Ocultar inicialmente
        
        tk.Label(self.motion_frame, text="Tamanho:", font=("Arial", 9)).grid(row=0, column=0, padx=5, sticky=tk.W)
        self.motion_size_var = tk.StringVar(value="20")
        motion_size_entry = tk.Entry(self.motion_frame, textvariable=self.motion_size_var, width=8, font=("Arial", 9))
        motion_size_entry.grid(row=0, column=1, padx=5)
        
        tk.Label(self.motion_frame, text="Comprimento:", font=("Arial", 9)).grid(row=0, column=2, padx=5, sticky=tk.W)
        self.length_var = tk.StringVar(value="10")
        length_entry = tk.Entry(self.motion_frame, textvariable=self.length_var, width=8, font=("Arial", 9))
        length_entry.grid(row=0, column=3, padx=5)
        
        tk.Label(self.motion_frame, text="Ângulo:", font=("Arial", 9)).grid(row=0, column=4, padx=5, sticky=tk.W)
        self.angle_var = tk.StringVar(value="0")
        angle_entry = tk.Entry(self.motion_frame, textvariable=self.angle_var, width=8, font=("Arial", 9))
        angle_entry.grid(row=0, column=5, padx=5)
        
        # Parâmetro de iterações (comum para ambos)
        tk.Label(params_frame, text="Iterações:", font=("Arial", 9)).grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.iterations_var = tk.StringVar(value="30")
        iterations_entry = tk.Entry(params_frame, textvariable=self.iterations_var, width=8, font=("Arial", 9))
        iterations_entry.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Botão executar
        self.execute_btn = tk.Button(
            control_frame,
            text="Executar",
            command=self.execute_deconvolution,
            font=("Arial", 11, "bold"),
            bg="#4CAF50",
            fg="white",
            padx=20,
            pady=8,
            state=tk.DISABLED
        )
        self.execute_btn.pack(side=tk.RIGHT, padx=10)
        
        # Frame principal para imagens e log
        main_content = tk.Frame(self.root)
        main_content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame para imagens
        images_frame = tk.Frame(main_content)
        images_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Frame para imagem original
        original_frame = tk.Frame(images_frame)
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        tk.Label(original_frame, text="Imagem Original", font=("Arial", 10, "bold")).pack(pady=5)
        self.original_canvas = tk.Canvas(original_frame, bg="gray90", highlightthickness=1, highlightbackground="gray")
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Frame para imagem deconvoluída
        deconvolved_frame = tk.Frame(images_frame)
        deconvolved_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        tk.Label(deconvolved_frame, text="Imagem Deconvoluída", font=("Arial", 10, "bold")).pack(pady=5)
        self.deconvolved_canvas = tk.Canvas(deconvolved_frame, bg="gray90", highlightthickness=1, highlightbackground="gray")
        self.deconvolved_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Frame para área de log
        log_frame = tk.Frame(main_content, width=300)
        log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5)
        log_frame.pack_propagate(False)
        
        tk.Label(log_frame, text="Log de Execução", font=("Arial", 10, "bold")).pack(pady=5)
        
        # Área de texto com scrollbar para logs
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            width=35,
            height=20,
            font=("Consolas", 9),
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="#ffffff",
            state=tk.DISABLED
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Botão para limpar log
        clear_log_btn = tk.Button(
            log_frame,
            text="Limpar Log",
            command=self.clear_log,
            font=("Arial", 8),
            padx=5,
            pady=2
        )
        clear_log_btn.pack(pady=5)
        
        # Label de status
        self.status_label = tk.Label(
            self.root,
            text="Pronto",
            font=("Arial", 9),
            fg="gray",
            anchor=tk.W
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        # Botão para salvar a imagem deconvoluída
        self.save_deconvolved_btn = tk.Button(
            control_frame,
            text="Salvar Imagem Deconvoluída",
            command=self.save_deconvolved_image,
            font=("Arial", 10),
            padx=10,
            pady=5,
            state=tk.DISABLED
        )
        self.save_deconvolved_btn.pack(side=tk.RIGHT, padx=5)
    
    def add_log_message(self, message):
        """Adiciona uma mensagem à área de log."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)  # Scroll para o final
        self.log_text.config(state=tk.DISABLED)
    
    def clear_log(self):
        """Limpa a área de log."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def on_blur_type_change(self, event=None):
        """Atualiza a interface quando o tipo de blur muda."""
        blur_type = self.blur_type_var.get()
        if blur_type == "gaussian":
            self.gaussian_frame.grid()
            self.motion_frame.grid_remove()
        else:  # motion
            self.gaussian_frame.grid_remove()
            self.motion_frame.grid()
    
    def select_image(self):
        """Abre diálogo para selecionar imagem."""
        file_path = filedialog.askopenfilename(
            title="Selecionar Imagem",
            filetypes=[
                ("Imagens", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("Todos os arquivos", "*.*")
            ]
        )
        
        if file_path:
            self.image_path = file_path
            # Extrair apenas o nome do arquivo (funciona em Windows e Linux)
            filename = os.path.basename(file_path)
            self.image_label.config(text=filename, fg="black")
            self.load_and_display_original()
            self.execute_btn.config(state=tk.NORMAL)
    
    def load_and_display_original(self):
        """Carrega e exibe a imagem original."""
        try:
            img = Image.open(self.image_path)
            if img.mode != 'RGB' and img.mode != 'L':
                img = img.convert('RGB')
            
            # Converter para array numpy e normalizar
            self.original_image = np.array(img, dtype=np.float64) / 255.0
            
            # Exibir imagem
            self.display_image(self.original_image, self.original_canvas)
            self.status_label.config(text="Imagem carregada com sucesso", fg="green")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar imagem: {e}")
            self.status_label.config(text=f"Erro: {e}", fg="red")
    
    def display_image(self, image_array, canvas):
        """Exibe uma imagem em um canvas."""
        # Garantir que os valores estão no range [0, 1]
        image_array = np.clip(image_array, 0, 1)
        
        # Converter para uint8
        image_array = (image_array * 255).astype(np.uint8)
        
        # Converter para PIL Image
        if len(image_array.shape) == 3:
            img = Image.fromarray(image_array, 'RGB')
        else:
            img = Image.fromarray(image_array, 'L')
        
        # Redimensionar para caber no canvas mantendo aspect ratio
        canvas.update_idletasks()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            img_width, img_height = img.size
            scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Converter para PhotoImage e exibir
        photo = ImageTk.PhotoImage(img)
        canvas.delete("all")
        canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo, anchor=tk.CENTER)
        canvas.image = photo  # Manter referência
    
    def execute_deconvolution(self):
        """Executa a deconvolução em uma thread separada."""
        if self.original_image is None:
            messagebox.showwarning("Aviso", "Por favor, selecione uma imagem primeiro.")
            return
        
        # Validar parâmetros
        try:
            blur_type = self.blur_type_var.get()
            algorithm_name = self.algorithm_var.get()
            
            if blur_type == "gaussian":
                size = int(self.size_var.get())
                sigma = float(self.sigma_var.get())
                if size <= 0 or sigma <= 0:
                    raise ValueError("Tamanho e sigma devem ser positivos")
            else:  # motion
                size = int(self.motion_size_var.get())
                length = float(self.length_var.get())
                angle = float(self.angle_var.get())
                if size <= 0 or length <= 0:
                    raise ValueError("Tamanho e comprimento devem ser positivos")
            
            iterations = int(self.iterations_var.get())
            if iterations <= 0:
                raise ValueError("Iterações devem ser positivas")
        except ValueError as e:
            messagebox.showerror("Erro", f"Parâmetros inválidos: {e}")
            return
        
        # Desabilitar botão durante processamento
        self.execute_btn.config(state=tk.DISABLED, text="Processando...")
        self.status_label.config(text=f"Processando deconvolução com {algorithm_name}...", fg="blue")
        
        # Limpar log anterior
        self.clear_log()
        self.add_log_message("=" * 50)
        self.add_log_message(f"Iniciando deconvolução com algoritmo: {algorithm_name}")
        self.add_log_message(f"Tipo de blur: {blur_type}")
        self.add_log_message("=" * 50)
        
        # Executar em thread separada para não travar a UI
        thread = threading.Thread(target=self._deconvolve_thread, args=(blur_type, algorithm_name))
        thread.daemon = True
        thread.start()

    def save_deconvolved_image(self):
        """Salva a imagem deconvoluída em um arquivo no diretório"""
        if self. deconvolved_image is None:
            return
        
        # Janela para escolher aonde salvar o arquivo
        file_path = filedialog.asksaveasfilename(
            title="Salvar Imagem Deconvoluída",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("BMP", "*.bmp"),
                ("TIFF", "*.tiff *.tif"),
                ("Todos os arquivos", "*.*")
            ]
        )

        if file_path:
            try:
                # Garantir que os valores estão no range [0, 1]
                image_array = np.clip(self.deconvolved_image, 0, 1)
                
                # Converter para uint8
                image_array = (image_array * 255).astype(np.uint8)
                
                # Converter para PIL Image
                if len(image_array.shape) == 3:
                    img = Image.fromarray(image_array, 'RGB')
                else:
                    img = Image.fromarray(image_array, 'L')
                
                # Salvar imagem
                img.save(file_path)
                messagebox.showinfo("Sucesso", f"Imagem deconvoluída salva em:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao salvar imagem: {e}")
    
    def _deconvolve_thread(self, blur_type, algorithm_name):
        """Executa a deconvolução em uma thread separada."""
        try:
            # Criar logger com callback para atualizar a UI
            logger = DeconvolutionLogger(callback=lambda msg: self.root.after(0, self.add_log_message, msg))
            
            # Gerar PSF
            logger.info(f"Gerando PSF do tipo '{blur_type}'...")
            if blur_type == "gaussian":
                size = int(self.size_var.get())
                sigma = float(self.sigma_var.get())
                psf = generate_gaussian_psf(size, sigma)
                logger.info(f"PSF gaussiana: tamanho={size}, sigma={sigma}")
            else:  # motion
                size = int(self.motion_size_var.get())
                length = float(self.length_var.get())
                angle = float(self.angle_var.get())
                psf = generate_motion_psf(size, length, angle)
                logger.info(f"PSF de movimento: tamanho={size}, comprimento={length}, ângulo={angle}°")
            
            # Aplicar deconvolução
            iterations = int(self.iterations_var.get())
            logger.info(f"Parâmetros: {iterations} iterações, clipping ativado")
            
            deconvolved = deconvolve(
                self.original_image,
                psf,
                algorithm_name=algorithm_name,
                num_iterations=iterations,
                clip=True,
                logger=logger
            )
            
            self.deconvolved_image = deconvolved
            
            # Atualizar UI na thread principal
            self.root.after(0, self._update_ui_after_deconvolution, True, None)
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda: self.add_log_message(f"ERRO: {error_msg}"))
            self.root.after(0, self._update_ui_after_deconvolution, False, error_msg)
    
    def _update_ui_after_deconvolution(self, success, error_msg):
        """Atualiza a UI após a deconvolução."""
        if success:
            self.display_image(self.deconvolved_image, self.deconvolved_canvas)
            self.status_label.config(text="Deconvolução concluída com sucesso!", fg="green")
            self.save_deconvolved_btn.config(state=tk.NORMAL)
        else:
            messagebox.showerror("Erro", f"Erro durante deconvolução: {error_msg}")
            self.status_label.config(text=f"Erro: {error_msg}", fg="red")
        
        self.execute_btn.config(state=tk.NORMAL, text="Executar")


def main():
    root = tk.Tk()
    app = DeconvolutionGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()

