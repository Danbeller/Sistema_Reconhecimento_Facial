# -*- coding: utf-8 -*-
"""
Sistema profissional de reconhecimento facial com OpenCV.

Motor de reconhecimento: LBPH (Local Binary Pattern Histograms)
  - Algoritmo nativo do OpenCV especificamente projetado para faces.
  - Muito superior ao ORB para este caso: analisa textura local do rosto
    e e robusto a variacoes de iluminacao e pequenas mudancas de expressao.

Requisito: opencv-contrib-python  (pip install opencv-contrib-python)

Fluxo:
  Cadastro : posicione o rosto no guia oval -> captura automatica de
             CAPTURAS_ALVO amostras -> treina/re-treina o modelo LBPH.
  Validacao: reconhecimento continuo em tempo real, sem tecla de acao.
             Exige FRAMES_CONFIRMAR frames consecutivos positivos antes
             de autorizar (elimina falsos positivos instantaneos).
             Anti-spoofing: verifica variacao temporal entre frames para
             rejeitar fotos impressas ou em tela.

Estrutura de arquivos:
  data/
    face_db.json          mapa label <-> nome
    lbph_model.xml        modelo treinado (LBPH)
    <nome>/               imagens de cada pessoa cadastrada
      <nome>_ts_N.png
"""

from pathlib import Path
import json
import time
import cv2
import numpy as np

# ─────────────────────────── constantes ────────────────────────────────────

DATA_DIR        = Path("data")
DB_PATH         = DATA_DIR / "face_db.json"
MODEL_PATH      = DATA_DIR / "lbph_model.xml"

FACE_SIZE       = (200, 200)   # tamanho normalizado de cada amostra
CAPTURAS_ALVO   = 25           # amostras coletadas por cadastro
INTERVALO_CAP   = 500          # ms entre capturas automaticas
MIN_FACE_PX     = 80           # tamanho minimo de face detectada (pixels)

# Confianca LBPH: 0 = identico, valores maiores = menos parecido.
# Abaixo de LBPH_THRESHOLD o rosto e reconhecido; acima, rejeitado.
LBPH_THRESHOLD  = 68.0

FRAMES_CONFIRMAR = 10          # frames consecutivos para confirmar identidade
LIVENESS_WINDOW  = 14          # frames para calculo de variacao temporal
LIVENESS_MIN_STD = 3.2         # desvio minimo para considerar rosto "vivo"

# Proporcao ideal: rosto deve ocupar entre estas fracoes da altura do frame
RATIO_MIN     = 0.22
RATIO_MAX     = 0.60
RATIO_OK_MIN  = 0.28
RATIO_OK_MAX  = 0.52


# ─────────────────────────── utilitarios de disco ──────────────────────────

def ensure_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)


def load_db() -> dict:
    """Retorna {label_int: nome_str}. Labels sao inteiros >= 0."""
    if DB_PATH.exists():
        raw = json.loads(DB_PATH.read_text(encoding="utf-8"))
        return {int(k): v for k, v in raw.items()}
    return {}


def save_db(label_map: dict) -> None:
    DB_PATH.write_text(
        json.dumps(
            {str(k): v for k, v in label_map.items()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def get_label_for(label_map: dict, nome: str) -> int:
    """Retorna label existente para o nome ou cria um novo."""
    for lbl, n in label_map.items():
        if n == nome:
            return lbl
    new_lbl = max(label_map.keys(), default=-1) + 1
    label_map[new_lbl] = nome
    return new_lbl


# ─────────────────────────── visao computacional ───────────────────────────

def get_detector() -> cv2.CascadeClassifier:
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    det = cv2.CascadeClassifier(path)
    if det.empty():
        raise RuntimeError(
            "Classificador Haar nao encontrado. Verifique a instalacao do OpenCV."
        )
    return det


def extract_face(frame: np.ndarray, detector: cv2.CascadeClassifier) -> tuple:
    """
    Detecta o maior rosto no frame.
    Retorna (face_gray, (x,y,w,h), gray_frame) ou (None, None, gray_frame).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=6,
        minSize=(MIN_FACE_PX, MIN_FACE_PX),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return None, None, gray
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return gray[y : y + h, x : x + w], (x, y, w, h), gray


def preprocess(face_gray: np.ndarray) -> np.ndarray:
    """
    Normaliza o rosto antes de treinar / comparar:
      1. Redimensiona para FACE_SIZE padrao.
      2. CLAHE: equaliza histograma de forma adaptativa por regiao,
         tornando o algoritmo robusto a diferentes iluminacoes.
    """
    resized = cv2.resize(face_gray, FACE_SIZE, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(resized)


# ─────────────────────────── modelo LBPH ───────────────────────────────────

def _create_recognizer() -> cv2.face.LBPHFaceRecognizer:
    return cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8,
        threshold=LBPH_THRESHOLD,
    )


def load_recognizer():
    """Carrega modelo salvo em disco. Retorna None se nao existir."""
    if not MODEL_PATH.exists():
        return None
    rec = _create_recognizer()
    rec.read(str(MODEL_PATH))
    return rec


def retrain_model(label_map: dict):
    """
    Re-treina o modelo LBPH do zero usando TODAS as imagens de TODOS os
    cadastros. Chamado apos cada novo cadastro para manter o modelo atualizado.
    Retorna o reconhecedor treinado ou None se nao houver imagens suficientes.
    """
    faces, labels = [], []
    for label, nome in label_map.items():
        person_dir = DATA_DIR / nome
        if not person_dir.exists():
            continue
        for img_path in sorted(person_dir.glob("*.png")):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(preprocess(img))
            labels.append(label)

    if len(faces) < 2:
        return None

    rec = _create_recognizer()
    rec.train(faces, np.array(labels, dtype=np.int32))
    return rec


# ─────────────────────────── anti-spoofing (liveness) ─────────────────────

class LivenessChecker:
    """
    Detecta se ha variacao temporal real entre frames consecutivos.
    Uma foto impressa ou exibida em tela produz frames quasi-identicos,
    resultando em variancia proxima de zero.
    Criterio: desvio padrao medio das diferencas absolutas entre frames
    consecutivos (64x64 px) deve superar LIVENESS_MIN_STD.
    """

    def __init__(self) -> None:
        self._buf: list = []

    def reset(self) -> None:
        self._buf.clear()

    def update(self, face_gray: np.ndarray) -> None:
        small = cv2.resize(face_gray, (64, 64)).astype(np.float32)
        self._buf.append(small)
        if len(self._buf) > LIVENESS_WINDOW:
            self._buf.pop(0)

    def is_live(self) -> bool:
        """True = rosto real  |  False = possivel foto ou tela."""
        if len(self._buf) < LIVENESS_WINDOW:
            return True  # ainda coletando — beneficio da duvida
        stds = [
            np.std(np.abs(self._buf[i + 1] - self._buf[i]))
            for i in range(len(self._buf) - 1)
        ]
        return float(np.mean(stds)) >= LIVENESS_MIN_STD


# ─────────────────────────── HUD / overlay ─────────────────────────────────

def _status_enquadramento(box: tuple, frame_h: int) -> tuple:
    """Retorna (mensagem, cor_BGR) conforme a proporcao do rosto no frame."""
    _, _, _, h = box
    ratio = h / frame_h
    if ratio < RATIO_MIN:
        return "APROXIME-SE", (0, 60, 255)
    if ratio > RATIO_MAX:
        return "AFASTE-SE", (0, 140, 255)
    if RATIO_OK_MIN <= ratio <= RATIO_OK_MAX:
        return "POSICAO IDEAL", (0, 210, 0)
    return "AJUSTE A DISTANCIA", (0, 200, 220)


def draw_hud(
    display: np.ndarray,
    box,
    total: int = 0,
    alvo: int = 0,
    modo: str = "validacao",
    resultado: str = "",
    res_color: tuple = (255, 255, 255),
) -> None:
    """
    Desenha HUD completo sobre o frame de exibicao:
      - Oval guia de enquadramento centralizada
      - Retangulo colorido ao redor do rosto detectado
      - Indicador de distancia textual
      - Barra de progresso (modo cadastro)
      - Resultado de validacao com fundo semitransparente
    """
    fh, fw = display.shape[:2]
    cx, cy = fw // 2, fh // 2

    # Oval guia — cinza discreto
    cv2.ellipse(
        display, (cx, cy), (fw // 5, int(fh * 0.38)),
        0, 0, 360, (160, 160, 160), 1,
    )

    if box is not None:
        x, y, w, h = box
        status, color = _status_enquadramento(box, fh)
        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            display, status, (10, fh - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA,
        )
    else:
        cv2.putText(
            display, "Nenhum rosto detectado", (10, fh - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 80, 255), 2, cv2.LINE_AA,
        )

    # ── Barra de progresso (cadastro) ──────────────────────────────────────
    if modo == "cadastro" and alvo > 0:
        bx, by, bh_bar = 10, 10, 20
        bw = fw - 20
        prog = int((total / alvo) * bw)
        cv2.rectangle(display, (bx, by), (bx + bw, by + bh_bar), (40, 40, 40), -1)
        cv2.rectangle(display, (bx, by), (bx + prog, by + bh_bar), (0, 200, 0), -1)
        cv2.rectangle(display, (bx, by), (bx + bw, by + bh_bar), (180, 180, 180), 1)
        cv2.putText(
            display, f"{total}/{alvo}",
            (bx + bw // 2 - 18, by + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA,
        )

    # ── Resultado de validacao ──────────────────────────────────────────────
    if modo == "validacao":
        cv2.putText(
            display, "Q = sair", (fw - 90, fh - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1, cv2.LINE_AA,
        )
        if resultado:
            overlay = display.copy()
            cv2.rectangle(overlay, (0, fh - 58), (fw, fh - 34), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)
            cv2.putText(
                display, resultado, (10, fh - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, res_color, 2, cv2.LINE_AA,
            )


# ─────────────────────────── fluxos principais ─────────────────────────────

def cadastrar_rosto(nome: str) -> None:
    """
    Captura CAPTURAS_ALVO amostras do rosto de forma totalmente automatica.
    A captura ocorre a cada INTERVALO_CAP ms quando o enquadramento e ideal.
    Apos concluir, re-treina o modelo LBPH com TODOS os cadastros existentes.
    """
    ensure_dirs()
    person_dir = DATA_DIR / nome
    person_dir.mkdir(parents=True, exist_ok=True)

    detector = get_detector()
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Webcam nao encontrada. Verifique a conexao.")

    label_map = load_db()
    get_label_for(label_map, nome)  # registra label no mapa

    total = 0
    last_cap_ms = 0

    print(f"\n  Cadastro de '{nome}'.")
    print(f"  Posicione o rosto dentro do guia oval e aguarde.")
    print(f"  Serao coletadas {CAPTURAS_ALVO} amostras automaticamente.")
    print("  ESC para cancelar.\n")

    while total < CAPTURAS_ALVO:
        ok, frame = cam.read()
        if not ok:
            continue

        fh = frame.shape[0]
        face, box, _ = extract_face(frame, detector)
        now_ms = int(time.time() * 1000)

        if face is not None and box is not None:
            status, _ = _status_enquadramento(box, fh)
            if status == "POSICAO IDEAL" and (now_ms - last_cap_ms) >= INTERVALO_CAP:
                processed = preprocess(face)
                img_path = person_dir / f"{nome}_{now_ms}_{total + 1}.png"
                cv2.imwrite(str(img_path), processed)
                total += 1
                last_cap_ms = now_ms
                print(f"  [{total:02d}/{CAPTURAS_ALVO}] Amostra capturada.")

        display = frame.copy()
        draw_hud(display, box, total=total, alvo=CAPTURAS_ALVO, modo="cadastro")
        cv2.imshow("Cadastro", display)

        if cv2.waitKey(1) & 0xFF == 27:
            print("  Cadastro cancelado.")
            cam.release()
            cv2.destroyAllWindows()
            return

    cam.release()
    cv2.destroyAllWindows()

    if total == 0:
        print("  Nenhuma amostra capturada.")
        return

    print("\n  Treinando modelo (aguarde)...")
    save_db(label_map)
    rec = retrain_model(label_map)
    if rec is None:
        print("  ERRO: imagens insuficientes para treinar o modelo.")
        return
    rec.write(str(MODEL_PATH))
    print(f"  Cadastro concluido. Modelo salvo em '{MODEL_PATH}'.\n")


def validar() -> None:
    """
    Validacao continua e automatica em tempo real.
    Seguranca em tres camadas:
      1. Confianca LBPH abaixo do limiar (LBPH_THRESHOLD).
      2. FRAMES_CONFIRMAR frames consecutivos positivos (anti-flash).
      3. Verificacao de liveness — rejeita fotos e telas sem variacao temporal.
    """
    label_map = load_db()
    if not label_map:
        print("  Banco vazio. Cadastre pelo menos um rosto antes de validar.")
        return

    rec = load_recognizer()
    if rec is None:
        print("  Modelo nao encontrado. Refaca o cadastro.")
        return

    detector  = get_detector()
    cam       = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Webcam nao encontrada. Verifique a conexao.")

    liveness      = LivenessChecker()
    consec: dict  = {}          # {label: contagem de frames positivos}
    resultado     = ""
    res_color     = (255, 255, 255)
    res_expira_ms = 0
    RESULT_TTL    = 3000        # ms que o resultado fica visivel

    print("\n  Validacao ativa. Olhe para a camera.")
    print("  Reconhecimento automatico continuo. Q para sair.\n")

    while True:
        ok, frame = cam.read()
        if not ok:
            continue

        face, box, _ = extract_face(frame, detector)
        now_ms = int(time.time() * 1000)

        if face is not None:
            liveness.update(face)
            processed = preprocess(face)

            try:
                label_pred, confidence = rec.predict(processed)
            except Exception:
                label_pred, confidence = -1, float("inf")

            if confidence <= LBPH_THRESHOLD and label_pred in label_map:
                consec[label_pred] = consec.get(label_pred, 0) + 1
                for k in list(consec.keys()):
                    if k != label_pred:
                        consec[k] = 0

                if consec[label_pred] >= FRAMES_CONFIRMAR:
                    consec[label_pred] = 0
                    if not liveness.is_live():
                        resultado     = "! ALERTA: POSSIVEL FOTO OU TELA !"
                        res_color     = (0, 0, 220)
                        res_expira_ms = now_ms + RESULT_TTL
                        liveness.reset()
                        print("  ALERTA: liveness reprovado — possivel ataque com foto.")
                    else:
                        nome = label_map[label_pred]
                        resultado     = f"AUTORIZADO: {nome}   [{confidence:.1f}]"
                        res_color     = (0, 210, 0)
                        res_expira_ms = now_ms + RESULT_TTL
                        liveness.reset()
                        print(f"  Autorizado: {nome} | confianca: {confidence:.1f}")
            else:
                consec.clear()
                # Exibe "nao reconhecido" apenas quando ha rosto mas confianca falhou
                if confidence != float("inf"):
                    resultado     = f"NAO RECONHECIDO   [{confidence:.1f}]"
                    res_color     = (0, 50, 220)
                    res_expira_ms = now_ms + RESULT_TTL
        else:
            consec.clear()
            liveness.reset()

        if now_ms > res_expira_ms:
            resultado = ""

        display = frame.copy()
        draw_hud(
            display, box,
            modo="validacao",
            resultado=resultado,
            res_color=res_color,
        )
        cv2.imshow("Validacao", display)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cam.release()
    cv2.destroyAllWindows()


# ─────────────────────────── interface ─────────────────────────────────────

def main() -> None:
    if not hasattr(cv2, "face"):
        print(
            "\n[ERRO] Modulo cv2.face nao encontrado.\n"
            "Instale com: pip install opencv-contrib-python\n"
            "Obs: nao instale opencv-python e opencv-contrib-python juntos.\n"
        )
        return

    print("\n╔══════════════════════════════════════════╗")
    print("║    SISTEMA DE RECONHECIMENTO FACIAL       ║")
    print("╠══════════════════════════════════════════╣")
    print("║  1 - Cadastrar novo rosto                 ║")
    print("║  2 - Validar rosto                        ║")
    print("║  3 - Sair                                 ║")
    print("╚══════════════════════════════════════════╝")
    opcao = input("  Escolha: ").strip()

    if opcao == "1":
        nome = input("  Nome (letras, numeros e _ apenas): ").strip()
        if not nome or not nome.replace("_", "").isalnum():
            print("  Nome invalido. Use apenas letras, numeros e underline.")
            return
        cadastrar_rosto(nome)
    elif opcao == "2":
        validar()
    elif opcao == "3":
        print("  Saindo.")
    else:
        print("  Opcao invalida.")


if __name__ == "__main__":
    main()
