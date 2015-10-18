#!/usr/bin/ruby

require "pp"
require "matrix"
require "csv"

class Player
  @learning_rate # 学習率(0,1]
  @discount_rate # 割引率[0,1]
  attr_reader :strategy      # 政策決定手段("man", "random", "greedy", "e-greedy", "roulette", "boltzman")
  attr_reader :q_table       # Q関数のテーブル 状態と行動を入力すると値（推定期待値）が得られる
  attr_reader :reward_count, :reward_sum # モンテカルロ法の報酬カウント/和テーブル

  @e_greedy_rate # e-greedy手法のランダム選択確率
  @templeture    # ボルツマン分布を使う時の温度パラメタ

  # コンストラクタ
  def initialize(strategy="random", learning_rate=0.1, discount_rate=0.9, e_greedy_rate=0.01, templeture=1.0)
    @strategy      = strategy
    @learning_rate = learning_rate
    @discount_rate = discount_rate
    @q_table       = Hash.new(0.to_f)
    @e_greedy_rate = e_greedy_rate
    @templeture    = templeture
    @reward_count  = Hash.new(0)
    @reward_sum    = Hash.new(0.to_f)
  end

  class StateAction
    attr_reader :state, :action

    def initialize(state, action)
      @state  = Marshal.load(Marshal.dump(state))
      @action = Marshal.load(Marshal.dump(action))
    end

    def ==(other)
      (@state == other.state && @action == other.action)
    end

    alias eql? ==

    def hash
      [@state, @action].hash
    end
  end

  # 状態とアクションの組からQ関数の値を得る. 
  def Qfunc(state, action)
    arg = StateAction.new(state, action)
    return @q_table[arg]
  end

  # Q関数の更新
  def set_Qfunc(state, action, value)
    arg = StateAction.new(state, action)
    @q_table[arg] = value
    # printf("q_table[%s]: %f \n", arg.to_s, @q_table[arg])
  end

  # 行動決定関数. 状態と行動可能なアクションを受け取って選んだアクション(整数)を返す
  def ActionSelect(state, playable_action)
    case @strategy
    when "man" then
      # 人間に選択させる
      return man_select(state, playable_action)
    when "random" then
      # 全くのランダム選択
      return playable_action.sample
    when "greedy" then
      # 一番Q値が高い行動を選択. ホモは欲張り
      return greedy_select(state, playable_action)
    when "e-greedy" then
      # e_greedy_rateの確率でランダム選択を行う
      return playable_action.sample if (rand < @e_greedy_rate)
      # それ以外はgreedyと同じ
      return greedy_select(state, playable_action)
    when "roulette" then
      # Q値の単純な比で確率分布を作り, その確率分布に従って選択
      return roulette_select(state, playable_action)
    when "boltzman" then
      # Q値の比でボルツマン分布を作り, その確率分布に従って選択
      return boltzman_select(state, playable_action)
    end
  end

  # Q関数の更新. 報酬を受け取ってQテーブルを更新する.
  def update_Q(state, action, after_state, after_playable_actions, reward)
    return nil if (@strategy == "man") or (@strategy == "random")
    q_prime_list = Array.new
    after_playable_actions.each do |after_action|
      q_prime_list.push(Qfunc(after_state, after_action))
    end
    # 更新
    update_q = (1-@learning_rate) * Qfunc(state, action) + @learning_rate * (reward + @discount_rate * q_prime_list.max)
    set_Qfunc(state, action, update_q)
  end

  # Sarsa方式のQ関数の更新. 
  # 上の更新と異なるのは, 後でのアクションが確定してから入力する
  def update_Q_Sarsa(state, action, after_state, after_action, reward)
    return nil if (@strategy == "man") or (@strategy == "random")
    q_prime  = Qfunc(after_state, after_action)
    update_q = (1-@learning_rate) * Qfunc(state, action) + @learning_rate * (reward + @discount_rate * q_prime)
    set_Qfunc(state, action, update_q)
  end

  # モンテカルロ法のQ関数の更新.
  # 報酬を手に入れるまでの状態,アクション配列を受け取る
  def update_Q_MonteCarlo(state_seq, action_seq, reward)
    return nil if (@strategy == "man") or (@strategy == "random")
    seq_length = action_seq.size
    for t in 0...seq_length do
      arg_t = StateAction.new(state_seq[t], action_seq[t])
      reward_count[arg_t] += 1
      reward_sum[arg_t]   += reward
      update_q = reward_sum[arg_t]/reward_count[arg_t]
      set_Qfunc(state_seq[t], action_seq[t], update_q)
    end
  end

  # 人間に選択させる
  def man_select(state, playable_action)
    while (true) do
      puts("Current state=")
      p state
      puts("Playable actions=")
      playable_action.each_with_index do |action, index|
        printf("[%d] %s ", index, action)
      end
      puts("")
      puts("Plesase input your action(integer):")
      select_str = STDIN.gets
      select_str.slice!("\n")
      select_index = select_str.to_i
      if (playable_action.include?(playable_action[select_index])) then
        return playable_action[select_index]
      else
        puts("Plesase select playable action(integer)")
      end
    end
  end

  # 状態と行動可能なアクションを受け取って最大のQ値を与えるアクションを返す
  def greedy_select(state, playable_action)
    q_list     = Array.new(playable_action.size)
    playable_action.each_with_index do |action, index|
      q_list[index] = Qfunc(state, action)
    end
    return playable_action[q_list.index(q_list.max)] if q_list.count(q_list.max) == 1
    # 最大のQ値を与えるアクションが複数あったら, ランダム選択
    max_q_actions = Array.new
    q_list.each_with_index do |q,inx|
      max_q_actions.push(playable_action[inx]) if q == q_list.max
    end
    return max_q_actions.sample
  end

  # ルーレット選択を行う. Q関数の比で確率分布を作って選択
  def roulette_select(state, playable_action)
    # 確率分布の作成
    prob_list    = Array.new(playable_action.size)
    sum_q        = 0.0
    playable_action.each_with_index do |action,index| 
      prob_list[index] = Qfunc(state, action) 
      sum_q           += prob_list[index]
    end

    # 負の値をとる場合, 最小値を引き, 全ての要素を正とする
    if prob_list.min < 0 then
      min = prob_list.min
      sum_q = 0.0
      prob_list.map! {|elm| elm -= min; sum_q += elm; elm}
    end

    return playable_action.sample if (sum_q < Float::EPSILON)
    prob_list.map! { |elm| (elm / sum_q) }
    # 乱択
    return playable_action[sampling_from_problist(*prob_list)]
  end

  # ボルツマン分布に従ったルーレット選択を行う
  def boltzman_select(state, playable_action)
    # 確率分布の作成
    prob_list    = Array.new(playable_action.size)
    sum_q        = 0.0
    playable_action.each_with_index do |action,index| 
      prob_list[index] = Math.exp(Qfunc(state, action)/@templeture)
      sum_q           += prob_list[index]
    end
    return playable_action.sample if (sum_q < Float::EPSILON)
    prob_list.map! { |elm| (elm / sum_q) }
    # 乱択
    return playable_action[sampling_from_problist(*prob_list)]
  end

  # 確率分布からの乱択を行い, 選ばれたリストのインデックスを返す
  def sampling_from_problist(*prob_list)
    random_value = rand
    acum_prob    = 0.0; acum_index = 0
    while (true) do
      if (acum_prob <= random_value and random_value < (acum_prob + prob_list[acum_index])) then
        return acum_index
      end
      acum_prob  += prob_list[acum_index]
      acum_index += 1
    end
    return acum_index
  end

  # Q関数テーブルをファイルに保存する
  def save_Qtable(filename)
    File.open(filename, 'w') {|f| f.write(Marshal.dump(@q_table)) }
  end

  # Q関数テーブルをファイルから読み込む
  def load_Qtable(filename)
    @q_table = Hash.new(0.to_f)
    @q_table = Marshal.load(File.read(filename))
  end

end

# 行列要素への代入
class Matrix
  def []=(i,j,x)
    @rows[i][j]=x
  end
end

class Othello_ja
  attr_reader :board # ゲーム盤
  @board_size        # 盤のサイズ
  # マスの状態
  STONE_EMPTY = 0   # 置かれていない
  STONE_BLACK = 1   # 黒が置かれている
  STONE_WHITE = 2   # 白が置かれている
  # マスの文字
  STONE_EMPTY_CHAR = "-"
  STONE_WHITE_CHAR = "@"
  STONE_BLACK_CHAR = "*"
  # 走査する八方向
  EIGHT_DIRECTION = [[-1,-1],[-1,0],[-1,1],[0,-1],[1,-1],[1,0],[0,1],[1,1]]
  attr_reader :player, :opponent
  attr_accessor :is_log
  attr_reader :player_score, :opponent_score
  attr_reader :player_win_count, :opponent_win_count
  @player_color
  @opponent_color
  @turn_count
  @pos_alphabet

  def initialize(board_size=4, player_strategy="man", opponent_strategy="random", is_log=false)
    @board_size = board_size
    @player     = Player.new(player_strategy)
    @opponent   = Player.new(opponent_strategy)
    @is_log     = is_log
    @player_win_count = 0
    @opponent_win_count = 0
    @pos_alphabet = ('a'..'z').to_a.first(@board_size)
  end

  def gameplay
    # 盤面の初期化
    @board                           = Matrix.zero(@board_size)
    harf_size = @board_size/2
    @board[harf_size-1, harf_size-1] = STONE_WHITE
    @board[harf_size, harf_size]     = STONE_WHITE
    @board[harf_size, harf_size-1]   = STONE_BLACK
    @board[harf_size-1, harf_size]   = STONE_BLACK

    # 色の確定
    if (rand < 0.5) then
      @player_color   = STONE_WHITE
      @opponent_color = STONE_BLACK
    else
      @opponent_color = STONE_WHITE
      @player_color   = STONE_BLACK
    end
    @player_score   = @board.count(@player_color)
    @opponent_score = @board.count(@opponent_color)

    # 先手は白とする
    turn_color = STONE_WHITE
    @turn_count = 0
    pass_count  = 0

    # モンテカルロ法用の状態, 行動列
    player_state_seq  = Array.new
    player_action_seq = Array.new
    opponent_state_seq  = Array.new
    opponent_action_seq = Array.new

    # ゲームループ
    while (true) do
      print_board if @is_log
      state = board_convert_to_int
      playable_action = get_playable_action(turn_color)
      unless (playable_action.size == 0) then
        pass_count = 0
        # アクション選択
        action = 
          if (@player_color == turn_color) and (@player.strategy == "man") then
            while (true) do
              puts("次の候補から, 手を整数で入力して<ENTER>を押してください")
              puts("(x,y)は盤面の座標を表しています:")
              playable_action.each_with_index do |pos,inx|
                printf("[%d]:(%c,%d) ", inx, @pos_alphabet[pos[0]], pos[1]+1)
              end
              puts("")
              input = STDIN.gets
              input.slice!("\n")
              input_int = input.to_i
              if (playable_action.include?(playable_action[input_int])) then
                break
              else
                puts("行動可能な整数を入力してください")
              end
            end
            playable_action[input_int]
          elsif (@player_color == turn_color) then
            @player.ActionSelect(state, playable_action)
          else
            if @is_log then
              printf("CPUのターン")
              # 10.times { sleep(0.1); printf(".") }
              printf("\n")
            end
            @opponent.ActionSelect(state, playable_action)
          end

        # 石を置く
        update_board(action[0], action[1], turn_color)

        # スコア（石の数）更新
        @player_score   = @board.count(@player_color)
        @opponent_score = @board.count(@opponent_color)

        # モンテカルロ法のために系列を保存
        if (turn_color == @player_color) then
          player_state_seq.push(Marshal.load(Marshal.dump(state)))
          player_action_seq.push(Marshal.load(Marshal.dump(action)))
        else
          opponent_state_seq.push(Marshal.load(Marshal.dump(state)))
          opponent_action_seq.push(Marshal.load(Marshal.dump(action)))
        end

      else
        # 置けない
        if @is_log then
          if turn_color == @player_color then
            puts("<<あなたはどこにも置けない...強制パス>>") 
          else
            puts("<<相手はどこにも置けない！>>") 
          end
        end
        pass_count += 1
        # 両者とも置けない場合はゲーム終了
        if (pass_count >= 2) then
          puts("<<両者とも置けない！ゲーム終了！>>") if @is_log
          break
        end
      end

      # ゲーム終了判定
      if (@player_score + @opponent_score) == @board_size**2
        break
      end

      # パーフェクトゲーム判定
      if (@player_score == 0) or (@opponent_score == 0) then
        puts("パーフェクトゲーム！！") if @is_log
        break
      end

      # ターンの交代
      @turn_count += 1
      turn_color = (turn_color == STONE_WHITE) ? STONE_BLACK : STONE_WHITE
    end

    # モンテカルロ法で学習
    @player.update_Q_MonteCarlo(player_state_seq, player_action_seq, @player_score)
    @opponent.update_Q_MonteCarlo(opponent_state_seq, opponent_action_seq, @opponent_score)

    # 勝敗判定
    print_board if @is_log
    if (@player_score > @opponent_score) then
      puts("あなたの勝ち！") if @is_log 
      @player_win_count += 1
    elsif (@player_score < @opponent_score) then
      puts("あなたの負け！") if @is_log 
      @opponent_win_count += 1
    else
      puts("引き分け！") if @is_log 
    end

  end

  # 盤面を一意な整数に変換する
  def board_convert_to_int
    # まず3進文字列を作り, それを10進に直す.
    @board.to_a.join.to_i(3)
  end

  # 盤面を整数から復元する
  def int_convert_to_board(code)
  end

  # 座標(x_pos,y_pos)で色colorの石を置けるかチェック
  def placeable?(x_pos, y_pos, color)
    return false if (@board[x_pos, y_pos] == STONE_WHITE) or (@board[x_pos, y_pos] == STONE_BLACK)
    another_color = (color == STONE_WHITE) ? STONE_BLACK : STONE_WHITE
    # 八方向夫々について置けるか調べる
    EIGHT_DIRECTION.each do |direction|
      tmp_pos = [x_pos+direction[0], y_pos+direction[1]]
      next if (@board[tmp_pos[0], tmp_pos[1]] == color) or (@board[tmp_pos[0], tmp_pos[1]] == STONE_EMPTY)
      # 方向を1つずつ伸ばしていき, 他の色の先にある自分の色を探す
      while (@board[tmp_pos[0], tmp_pos[1]] == another_color) and check_pos(tmp_pos[0], tmp_pos[1]) do
        tmp_pos[0] += direction[0]
        tmp_pos[1] += direction[1]
      end
      return true if (@board[tmp_pos[0], tmp_pos[1]] == color) and check_pos(tmp_pos[0], tmp_pos[1])
    end
    
    return false
  end

  # 色colorの置ける座標配列を返す
  def get_playable_action(color)
    playable_action = Array.new
    @board.each_with_index do |e, x_pos, y_pos|
      playable_action.push([x_pos, y_pos]) if placeable?(x_pos, y_pos, color)
    end
    return playable_action
  end

  # (x_pos, y_pos)に色colorの石を置き, ひっくり返して盤面を更新
  # 置けるかどうかの検査はしない
  def update_board(x_pos, y_pos, color)
    pos_list = Array.new           # 色colorに変える座標リスト
    pos_list.push([x_pos, y_pos].clone)
    EIGHT_DIRECTION.each do |direction|
      tmp_pos      = [x_pos+direction[0], y_pos+direction[1]]
      next if (@board[tmp_pos[0], tmp_pos[1]] == color) or (@board[tmp_pos[0], tmp_pos[1]] == STONE_EMPTY)
      tmp_pos_list = Array.new
      another_color = (color == STONE_WHITE) ? STONE_BLACK : STONE_WHITE
      # 方向を1つずつ伸ばしていき, 他の色の先にある自分の色を探す
      while (@board[tmp_pos[0], tmp_pos[1]] == another_color) and check_pos(tmp_pos[0], tmp_pos[1]) do 
        tmp_pos_list.push(tmp_pos.clone)
        tmp_pos[0] += direction[0]
        tmp_pos[1] += direction[1]
      end
      if (@board[tmp_pos[0], tmp_pos[1]] == color) and check_pos(tmp_pos[0], tmp_pos[1]) then
        pos_list += Marshal.load(Marshal.dump(tmp_pos_list))
      end
    end
    # 一気に色を書き換える
    pos_list.each do |pos|
      @board[pos[0], pos[1]] = color
    end
  end

  # (x_pos, y_pos)が盤面から飛び出ていないかチェックする
  def check_pos(x_pos, y_pos)
    (x_pos < @board_size) and (x_pos >= 0) and (y_pos < @board_size) and (y_pos >= 0)
  end

  # 盤面を印字する
  def print_board
    # printf("\033[2J") # 画面フラッシュ
    printf("[ターンNo. %d]: \n", @turn_count)
    printf("    ")
    @board_size.times { |pos| printf("%c   ", @pos_alphabet[pos]) }
    printf("\n")
    printf("  ")
    (4 * @board_size + 1).times { printf("-") }
    printf("\n")
    for col in 0...@board_size do
      printf("%d |", col+1)
      @board.column(col) do |elm|
        case elm
        when STONE_EMPTY
          ch = STONE_EMPTY_CHAR
        when STONE_WHITE
          ch = STONE_WHITE_CHAR
        when STONE_BLACK
          ch = STONE_BLACK_CHAR
        end
        printf(" %s ", ch)
        printf("|")
      end
      printf("\n")
      printf("  ")
      (4 * @board_size + 1).times { printf("-") }
      printf("\n")
    end

    if (@player_color == STONE_WHITE) then
      player_stone_chr = STONE_WHITE_CHAR
      opponent_stone_chr = STONE_BLACK_CHAR
    else
      player_stone_chr = STONE_BLACK_CHAR
      opponent_stone_chr = STONE_WHITE_CHAR
    end
    # スコアの状態
    printf("プレイヤー[%s]の得点: %d, 相手[%s]の得点: %d\n", player_stone_chr, @player_score, opponent_stone_chr, @opponent_score)
    (40).times { printf("=") }
    printf("\n")
  end

end

class GameSimulator
  attr_reader :game
  @total_player_score
  @max_player_score
  @total_opponent_score
  @log_count

  def initialize(board_size, player_strategy, opponent_strategy, is_log)
    @game =  Othello_ja.new(board_size, player_strategy, opponent_strategy, is_log)
    @total_player_score   = 0
    @total_opponent_score = 0
    @log_count            = 10 # ログの回数
    @max_player_score     = -Float::MAX
  end

  def simulation(max_cycle)
    pre_player_win = pre_opponent_win = 0
    pre_player_score = pre_opponent_score = 0
    1.upto(max_cycle) do |iteration|
      @game.gameplay
      @total_player_score += @game.player_score.to_f
      @total_opponent_score += @game.opponent_score.to_f

      # 学習状態の印字
      puts("[iteration:" + iteration.to_s + "] " + "Player win:" + (@game.player_win_count.to_f*100/iteration).to_s + "[%], avg. score:" + (@total_player_score/iteration).to_s + " vs. Opponent win:" + ((@game.opponent_win_count).to_f*100/iteration).to_s + "[%], avg. score:" + (@total_opponent_score/iteration).to_s + " Q-table size:" + @game.player.q_table.size.to_s) if (iteration % (max_cycle/@log_count)) == 0
    end
  end
end

def wait_for_play
  printf("何かキーを押してね")
  dum = STDIN.gets
  printf("\033[2J")
end

# シュミレーション
sim = GameSimulator.new(4, "e-greedy", "random", false)
sim.simulation(10000)
