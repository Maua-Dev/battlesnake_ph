
# Bem-vindo ao
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \   __\   __\  | _/ __ \ /  ___//    \__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\_____>
#
# ESTE É O ARQUIVO QUE VOCÊ VAI EDITAR. Todo o resto do projeto existe
# só para levar o estado do jogo até as quatro funções (info, start, end, get_move).
#
# Aqui a decisão de movimento usa Dijkstra (classe SnakeBrain, mais abaixo).
# Se algo der errado no cérebro, get_move cai num plano B simples e seguro.
# Documentação: https://docs.battlesnake.com
 
import heapq
import random
import logging
from collections import deque
from .models import GameState, MoveResponse
 
logger = logging.getLogger(__name__)
# O runtime Python da Lambda deixa o logger raiz em WARNING: sem esta linha
# as jogadas nao aparecem no CloudWatch.
logger.setLevel(logging.INFO)
 
 
# =====================================================================
#  CÉREBRO (Dijkstra + flood fill) - sem dependências externas
#  Coordenadas do Battlesnake: origem embaixo à esquerda, "up" = y + 1.
# =====================================================================
MOVES = ["up", "down", "left", "right"]
DX = [0, 0, -1, 1]
DY = [1, -1, 0, 0]
INF = float("inf")
CUSTO_HAZARD = 5  # entrar em casa de hazard custa mais no Dijkstra
 
 
# ---------- modelo de entrada (preencha a partir do JSON do /move) ----------
class Snk:
    def __init__(self, id="", health=100, body=None):
        self.id = id
        self.health = health
        self.body = body or []  # lista de (x, y); body[0] = cabeça
 
 
class State:
    def __init__(self, w=11, h=11, me=0, wrapped=False, food=None, hazards=None, snakes=None):
        self.w, self.h, self.me = w, h, me  # me = índice da sua cobra em snakes
        self.wrapped = wrapped              # ruleset "wrapped"
        self.food = food or []
        self.hazards = hazards or []
        self.snakes = snakes or []
 
 
class SnakeBrain:
    def __init__(self, s):
        self.s = s
        self.W, self.H = s.w, s.h
        self.V = self.W * self.H
        self.bloq = [False] * self.V
        self.perigo = [False] * self.V
        self.hazard = [False] * self.V
        self.tem_food = [False] * self.V
        for f in s.food:
            self.tem_food[self.cell(f)] = True
        for z in s.hazards:
            self.hazard[self.cell(z)] = True
        self.marca_obstaculos()
 
    def cell(self, p):
        return p[1] * self.W + p[0]
 
    def nb(self, c, d):
        """Vizinho na direção d, ou -1 se sair do tabuleiro."""
        x = c % self.W + DX[d]
        y = c // self.W + DY[d]
        if self.s.wrapped:
            x %= self.W
            y %= self.H
        elif x < 0 or y < 0 or x >= self.W or y >= self.H:
            return -1
        return y * self.W + x
 
    def marca_obstaculos(self):
        s = self.s
        me = s.snakes[s.me]
        for i, sn in enumerate(s.snakes):
            b = sn.body
            n = len(b)
            vai_comer = False
            if i != s.me:
                for d in range(4):
                    c = self.nb(self.cell(b[0]), d)
                    if c >= 0 and self.tem_food[c]:
                        vai_comer = True
            for k in range(n):
                cauda_solta = (k == n - 1 and n > 1 and tuple(b[n - 1]) != tuple(b[n - 2]) and not vai_comer)
                if not cauda_solta:
                    self.bloq[self.cell(b[k])] = True
            # cabeças de rivais maiores ou iguais: casas onde elas podem entrar são perigosas
            if i != s.me and n >= len(me.body):
                for d in range(4):
                    c = self.nb(self.cell(b[0]), d)
                    if c >= 0:
                        self.perigo[c] = True
 
    # ================= DIJKSTRA =================
    # Custo de ENTRAR na casa destino (1, ou CUSTO_HAZARD em hazard).
    # Usa heap + vizinhos calculados na hora (em Python, matriz VxV seria lenta).
    def dijkstra(self, src, evita_perigo):
        dist = [INF] * self.V
        pai = [-1] * self.V
        dist[src] = 0
        heap = [(0, src)]
        while heap:
            du, u = heapq.heappop(heap)
            if du > dist[u]:
                continue
            for d in range(4):
                b = self.nb(u, d)
                if b < 0 or b == u or self.bloq[b] or (evita_perigo and self.perigo[b]):
                    continue
                nd = du + (CUSTO_HAZARD if self.hazard[b] else 1)
                if nd < dist[b]:
                    dist[b] = nd
                    pai[b] = u
                    heapq.heappush(heap, (nd, b))
        return dist, pai
    # ============================================
 
    def espaco(self, start):
        """Flood fill: quantas casas livres alcançáveis."""
        vis = [False] * self.V
        q = deque([start])
        vis[start] = True
        n = 0
        while q:
            c = q.popleft()
            n += 1
            for d in range(4):
                b = self.nb(c, d)
                if b >= 0 and not vis[b] and not self.bloq[b]:
                    vis[b] = True
                    q.append(b)
        return n
 
    @staticmethod
    def primeiro_passo(pai, head, alvo):
        """Volta pelo vetor pai até o vizinho da cabeça."""
        c = alvo
        while c != -1 and pai[c] != head:
            c = pai[c]
        return c
 
    @staticmethod
    def dir_para(destino, to):
        if destino < 0:
            return -1
        for d in range(4):
            if to[d] == destino:
                return d
        return -1
 
    def decide(self):
        s = self.s
        me = s.snakes[s.me]
        head = self.cell(me.body[0])
        n = len(me.body)
        to = [self.nb(head, d) for d in range(4)]
        legal = [d for d in range(4) if to[d] >= 0 and not self.bloq[to[d]]]
        if not legal:
            return "up"  # sem saída
 
        safe = [d for d in legal if not self.perigo[to[d]]]
        if not safe:
            safe = legal
        esp = [0] * 4
        good = []
        for d in safe:
            esp[d] = self.espaco(to[d])
            if esp[d] >= n:
                good.append(d)
 
        # distâncias dos rivais (sem evitar perigo) e minhas (evitando perigo)
        dist_e = {}
        for j, sn in enumerate(s.snakes):
            if j != s.me:
                dist_e[j] = self.dijkstra(self.cell(sn.body[0]), False)[0]
        dist, pai = self.dijkstra(head, True)
 
        # 1) fruta que eu alcanço antes dos rivais (empate: só se eu for maior)
        alvo, melhor = -1, INF
        mais_perto, d_perto = -1, INF
        for f in s.food:
            c = self.cell(f)
            if dist[c] == INF:
                continue
            if dist[c] < d_perto:
                d_perto, mais_perto = dist[c], c
            disputada = False
            for j, sn in enumerate(s.snakes):
                if j == s.me:
                    continue
                dj, lj = dist_e[j][c], len(sn.body)
                if dj < dist[c] or (dj == dist[c] and lj >= n):
                    disputada = True
            if not disputada and dist[c] < melhor:
                melhor, alvo = dist[c], c
        if alvo == -1 and me.health <= 35:
            alvo = mais_perto  # com fome, arrisca a mais próxima
        if alvo != -1:
            d = self.dir_para(self.primeiro_passo(pai, head, alvo), to)
            if d >= 0 and d in good:
                return MOVES[d]
 
        # 2) perseguir a própria cauda
        if n > 1:
            tail = self.cell(me.body[-1])
            if not self.bloq[tail] and dist[tail] != INF:
                d = self.dir_para(self.primeiro_passo(pai, head, tail), to)
                if d >= 0 and d in good:
                    return MOVES[d]
 
        # 3) maior espaço livre
        pool = good if good else safe
        best = pool[0]
        for d in pool:
            if esp[d] > esp[best] or (esp[d] == esp[best] and not self.hazard[to[d]] and self.hazard[to[best]]):
                best = d
        return MOVES[best]
 
 
def move(state):
    """Ponto de entrada para o handler (/move)."""
    return SnakeBrain(state).decide()
 
 
# =====================================================================
#  ADAPTADOR: GameState (models.py) -> State do cérebro
# =====================================================================
def _xy(p):
    """Aceita objeto com .x/.y ou dict {"x":..,"y":..}."""
    if isinstance(p, dict):
        return (p["x"], p["y"])
    return (p.x, p.y)
 
 
def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
 
 
def _para_state(state: GameState) -> State:
    board = state.board
    snakes, me = [], -1
    my_id = _get(state.you, "id")
    for i, sn in enumerate(board.snakes):
        snakes.append(Snk(_get(sn, "id"), _get(sn, "health", 100), [_xy(p) for p in sn.body]))
        if _get(sn, "id") == my_id:
            me = i
    if me == -1:  # plano B: acha pela posição da cabeça
        head = _xy(state.you.body[0])
        for i, sn in enumerate(snakes):
            if sn.body and sn.body[0] == head:
                me = i
    if me == -1:  # a própria cobra não veio em board.snakes: adiciona
        snakes.append(Snk(my_id, _get(state.you, "health", 100), [_xy(p) for p in state.you.body]))
        me = len(snakes) - 1
 
    ruleset = _get(_get(state, "game"), "ruleset")
    wrapped = _get(ruleset, "name") == "wrapped"
    return State(
        board.width, board.height, me, wrapped,
        [_xy(p) for p in (_get(board, "food") or [])],
        [_xy(p) for p in (_get(board, "hazards") or [])],
        snakes,
    )
 
 
def _plano_b(state: GameState) -> str:
    """Regras simples do template: não voltar, não sair do tabuleiro, não bater em corpos."""
    try:
        head = _xy(state.you.body[0])
        ocupado = {_xy(p) for sn in state.board.snakes for p in sn.body}
        ocupado |= {_xy(p) for p in state.you.body}
        seguras = []
        for nome, d in zip(MOVES, range(4)):
            x, y = head[0] + DX[d], head[1] + DY[d]
            if 0 <= x < state.board.width and 0 <= y < state.board.height and (x, y) not in ocupado:
                seguras.append(nome)
        if seguras:
            return random.choice(seguras)
    except Exception:
        pass
    return random.choice(MOVES)
 
 
def info() -> dict:
    """GET / — chamado quando você cadastra a cobra e a cada partida.
    Controla a aparência dela.
    Opções de cabeça, cauda e cor: https://docs.battlesnake.com/guides/customizations
    """
    logger.info("INFO")
 
    return {
        "apiversion": "1",
        "author": "",          # TODO: coloque aqui o SEU usuário do Battlesnake
        "color": "#8B0000",    # TODO: escolha a cor da sua cobra
        "head": "tiger-king",  # TODO: escolha a cabeça
        "tail": "hook",        # TODO: escolha a cauda
        "version": "1.0.0",
    }
 
 
def start(state: GameState) -> None:
    """POST /start — chamado uma vez, quando a partida começa."""
    logger.info("JOGO COMEÇOU (partida %s)", state.game.id)
 
 
def end(state: GameState) -> None:
    """POST /end — chamado uma vez, quando a partida termina."""
    logger.info("FIM DE JOGO após %d turnos", state.turn)
 
 
def get_move(state: GameState) -> MoveResponse:
    """POST /move — chamado a cada turno. Devolve "up", "down", "left" ou "right"."""
    try:
        chosen = move(_para_state(state))
        logger.debug("MOVE %d: %s", state.turn, chosen)
    except Exception as e:  # nunca deixar estourar: cai no plano B
        chosen = _plano_b(state)
        logger.error("MOVE %d: erro no cérebro (%r) -> plano B: %s", state.turn, e, chosen)
    return MoveResponse(move=chosen)