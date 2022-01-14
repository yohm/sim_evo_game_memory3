# 16.times do |i|
#   out = `./cmake-build-release/test_StrategyM3 m1-#{i}`
#   print i, out.lines.grep(/IsDefensible: \d/), "\n"
# end

require 'rational'

def rationalize_payoff(line)
  if line =~ /(\d\.\d+) b - (\d\.\d+) c/
    r1 = Rational($1).rationalize(0.01)
    r2 = Rational($2).rationalize(0.01)
    return r1, r2
  else
    raise "invalid format"
  end
end

N = 16
pi_matrix = Array.new(N) {|i| Array.new(N) }

N.times do |i|
  N.times do |j|
    out = `./cmake-build-release/test_StrategyM3 m1-#{i} m1-#{j}`
    # line : "Payoff-1: 1.00 b - 1.00 c\n"
    l1 = out.lines.grep(/Payoff-1:/).first
    pi_matrix[i][j] = rationalize_payoff(l1)
  end
end

q_matrix = Array.new(N) {|i| Array.new(N) }
N.times do |i|
  N.times do |j|
    q0 = pi_matrix[i][j][0] - pi_matrix[j][i][0] + pi_matrix[i][i][0] - pi_matrix[j][j][0]
    q1 = pi_matrix[i][j][1] - pi_matrix[j][i][1] + pi_matrix[i][i][1] - pi_matrix[j][j][1]
    q_matrix[i][j] = [q0, q1]
  end
end

pp pi_matrix[0][1], pi_matrix[1][0], pi_matrix[0][0], pi_matrix[1][1]
pp q_matrix[0][1]

def print_matrix(matrix)
  puts N.times.map {|j| " #{j} "}.join("|")
  puts N.times.map {|j| " --- "}.join("|")
  N.times do |i|
    print "| #{i} "
    N.times do |j|
      r1,r2 = matrix[i][j]
      r1 = r1.to_i if r1 == 0 or r1 == 1
      r2 = r2.to_i if r2 == 0 or r2 == 1
      print " | $#{r1} b - #{r2} c$"
    end
    puts " | "
  end
end

def print_sign(matrix)
  puts N.times.map {|j| " #{j} "}.join("|")
  puts N.times.map {|j| " --- "}.join("|")
  N.times do |i|
    print "| #{i} "
    N.times do |j|
      r1,r2 = matrix[i][j]
      # Q_ij = r_1 b - r_2 c
      if r1 == 0
        if r2 == 0
          print " | 0    "
        elsif r2 > 0
          print " | -    "
        elsif r2 < 0
          print " | +    "
        end
      elsif r1 > 0
        b_th = r2 / r1
        if b_th > 1
          print " | b > #{b_th}c"
        else
          print " | +            "
        end
      elsif r1 < 0
        b_th = r2 / r1
        if b_th > 1
          print " | b < #{b_th}c"
        else
          print " | -            "
        end
      end
    end
    puts " | "
  end
end
print_sign(q_matrix)