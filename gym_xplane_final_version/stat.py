import pstats

# profile.pstats 파일 열기
stats = pstats.Stats('profile.pstats')

# 빈번한 호출 순으로 정렬된 통계 출력
stats.strip_dirs().sort_stats('calls').print_stats()