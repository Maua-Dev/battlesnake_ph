# Bem-vindo ao
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \   __\   __\  | _/ __ \ /  ___//    \__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\_____>
#
# Versão SIMPLES e rápida. Cada decisão é uma função pequena:
#   1. movimentos_seguros -> nunca bate na parede nem em corpos
#   2. caminho_ate_comida -> vai até a comida mais próxima (BFS)
#   3. espaco_livre       -> evita becos sem saída
# Documentação: https://docs.battlesnake.com

import logging
import random
from collections import deque

from .models import GameState, MoveResponse

logger = logging.getLogger(__name__)
# Sem esta linha as jogadas não aparecem no CloudWatch (logger raiz da Lambda = WARNING).
logger.setLevel(logging.INFO)

DIRECOES = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}


# --------------------------------------------------------------------------- #
# INFO / START / END
# --------------------------------------------------------------------------- #

def info() -> dict:
    return {
        "apiversion": "1",
        "author": "",          # TODO: coloque aqui o SEU usuário do Battlesnake
        "color": "#8B0000",
        "head": "tiger-king",
        "tail": "hook",
        "version": "2.0.0",
    }


def start(state: GameState) -> None:
    logger.info("JOGO COMEÇOU (partida %s)", state.game.id)


def end(state: GameState) -> None:
    logger.info("FIM DE JOGO após %d turnos", state.turn)


# --------------------------------------------------------------------------- #
# FUNÇÕES AUXILIARES
# --------------------------------------------------------------------------- #

def modo_wrapped(state: GameState) -> bool:
    """No modo 'wrapped' a parede não mata: a cobra aparece do outro lado."""
    ruleset = state.game.ruleset or {}
    return str(ruleset.get("name", "")).lower() == "wrapped"


def proxima_casa(state: GameState, pos, direcao):
    """Casa para onde o movimento leva. Devolve None se for PAREDE (fora do tabuleiro)."""
    dx, dy = DIRECOES[direcao]
    x, y = pos[0] + dx, pos[1] + dy
    largura, altura = state.board.width, state.board.height
    if modo_wrapped(state):
        return (x % largura, y % altura)
    if 0 <= x < largura and 0 <= y < altura:
        return (x, y)
    return None


def vizinhas(state: GameState, pos):
    """Casas vizinhas dentro do tabuleiro."""
    for direcao in DIRECOES:
        casa = proxima_casa(state, pos, direcao)
        if casa is not None:
            yield casa


def casas_bloqueadas(state: GameState) -> set:
    """Todas as casas ocupadas por corpos. A cauda conta como livre (ela sai do lugar),
    exceto se a cobra acabou de comer ou está colada numa comida (pode comer agora)."""
    comidas = {(c.x, c.y) for c in state.board.food}
    meu_id = state.you.id
    bloqueadas = {(c.x, c.y) for c in state.you.body}  # garante o meu corpo mesmo se faltar na lista

    for cobra in state.board.snakes:
        corpo = [(c.x, c.y) for c in cobra.body]
        cauda_solta = len(corpo) > 1 and corpo[-1] != corpo[-2]
        if cobra.id != meu_id and any(v in comidas for v in vizinhas(state, corpo[0])):
            cauda_solta = False  # rival pode comer: a cauda dele não sai do lugar
        for i, parte in enumerate(corpo):
            if i == len(corpo) - 1 and cauda_solta:
                continue
            bloqueadas.add(parte)

    # minha própria cauda também libera (se eu não acabei de comer)
    meu_corpo = [(c.x, c.y) for c in state.you.body]
    if len(meu_corpo) > 1 and meu_corpo[-1] != meu_corpo[-2]:
        so_minha_cauda = all(meu_corpo[-1] != (c.x, c.y) for s in state.board.snakes if s.id != meu_id for c in s.body)
        if so_minha_cauda:
            bloqueadas.discard(meu_corpo[-1])
    return bloqueadas


def casas_perigosas(state: GameState) -> set:
    """Casas onde uma cobra MAIOR ou IGUAL pode chegar no próximo turno (perderíamos o choque de cabeças)."""
    perigosas = set()
    meu_tamanho = len(state.you.body)
    for cobra in state.board.snakes:
        if cobra.id == state.you.id or len(cobra.body) < meu_tamanho:
            continue
        perigosas.update(vizinhas(state, (cobra.body[0].x, cobra.body[0].y)))
    return perigosas


def movimentos_seguros(state: GameState, bloqueadas: set, perigosas: set) -> list:
    """Direções que NÃO batem na parede nem em corpo. Prefere as que evitam choque de cabeças."""
    cabeca = (state.you.body[0].x, state.you.body[0].y)
    livres = []
    for direcao in DIRECOES:
        casa = proxima_casa(state, cabeca, direcao)
        if casa is not None and casa not in bloqueadas:
            livres.append(direcao)
    sem_perigo = [d for d in livres if proxima_casa(state, cabeca, d) not in perigosas]
    return sem_perigo or livres


def espaco_livre(state: GameState, inicio, bloqueadas: set, limite: int) -> int:
    """Quantas casas livres dá para alcançar a partir de 'inicio' (para no 'limite')."""
    visitadas = {inicio}
    fila = deque([inicio])
    while fila and len(visitadas) < limite:
        atual = fila.popleft()
        for viz in vizinhas(state, atual):
            if viz not in visitadas and viz not in bloqueadas:
                visitadas.add(viz)
                fila.append(viz)
    return len(visitadas)


def caminho_ate_comida(state: GameState, bloqueadas: set, evitar: set):
    """BFS a partir da cabeça. Devolve (direção do 1º passo, distância, casa da comida)
    para a comida mais próxima que nenhum rival MAIOR/IGUAL alcança antes. None se não houver."""
    cabeca = (state.you.body[0].x, state.you.body[0].y)
    comidas = {(c.x, c.y) for c in state.board.food}
    if not comidas:
        return None

    meu_tamanho = len(state.you.body)
    rivais = [(c.body[0].x, c.body[0].y) for c in state.board.snakes
              if c.id != state.you.id and len(c.body) >= meu_tamanho]

    primeiro = {cabeca: None}
    distancia = {cabeca: 0}
    fila = deque([cabeca])
    while fila:
        atual = fila.popleft()
        if atual in comidas and atual != cabeca:
            # rival maior/igual chega antes? (distância em linha reta, aproximação barata)
            perde = any(_manhattan(state, r, atual) < distancia[atual] for r in rivais)
            if not perde:
                return primeiro[atual], distancia[atual], atual
        for direcao in DIRECOES:
            viz = proxima_casa(state, atual, direcao)
            if viz is None or viz in distancia or viz in bloqueadas or viz in evitar:
                continue
            distancia[viz] = distancia[atual] + 1
            primeiro[viz] = primeiro[atual] or direcao
            fila.append(viz)
    return None


def _manhattan(state: GameState, a, b) -> int:
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    if modo_wrapped(state):
        dx, dy = min(dx, state.board.width - dx), min(dy, state.board.height - dy)
    return dx + dy


def distancia_do_centro(state: GameState, casa) -> float:
    cx, cy = (state.board.width - 1) / 2, (state.board.height - 1) / 2
    return abs(casa[0] - cx) + abs(casa[1] - cy)


def movimento_de_emergencia(state: GameState) -> str:
    """Tudo está bloqueado: escolhe qualquer movimento que ao menos não saia do tabuleiro."""
    cabeca = (state.you.body[0].x, state.you.body[0].y)
    dentro = [d for d in DIRECOES if proxima_casa(state, cabeca, d) is not None]
    return random.choice(dentro) if dentro else "up"


# --------------------------------------------------------------------------- #
# DECISÃO
# --------------------------------------------------------------------------- #

def escolher_movimento(state: GameState) -> str:
    cabeca = (state.you.body[0].x, state.you.body[0].y)
    tamanho = len(state.you.body)
    bloqueadas = casas_bloqueadas(state)
    perigosas = casas_perigosas(state)

    seguros = movimentos_seguros(state, bloqueadas, perigosas)
    if not seguros:
        return movimento_de_emergencia(state)

    # descarta becos: movimentos que deixam menos espaço do que o tamanho do corpo
    espaco = {d: espaco_livre(state, proxima_casa(state, cabeca, d), bloqueadas, limite=2 * tamanho + 10)
              for d in seguros}
    bons = [d for d in seguros if espaco[d] >= tamanho] or seguros

    # 1) comer: segue o caminho até a comida mais próxima (se for uma jogada boa)
    rota = caminho_ate_comida(state, bloqueadas, evitar=perigosas)
    if rota is not None and rota[0] in bons:
        return rota[0]

    # 2) sem comida alcançável: com fome, tenta a mais próxima sem evitar perigo
    if rota is None and state.you.health <= 40:
        rota = caminho_ate_comida(state, bloqueadas, evitar=set())
        if rota is not None and rota[0] in bons:
            return rota[0]

    # 3) senão: o movimento com mais espaço; empate -> mais perto do centro (longe da parede)
    return max(bons, key=lambda d: (espaco[d], -distancia_do_centro(state, proxima_casa(state, cabeca, d)),
                                    random.random()))


def get_move(state: GameState) -> MoveResponse:
    try:
        movimento = escolher_movimento(state)
        logger.info("MOVE %d: %s", state.turn, movimento)
    except Exception:  # nunca deixar a API falhar: um movimento qualquer é melhor que nenhum
        logger.exception("MOVE: erro na lógica, usando emergência")
        try:
            movimento = movimento_de_emergencia(state)
        except Exception:
            movimento = "up"
    return MoveResponse(move=movimento)