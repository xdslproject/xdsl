# RUN: filecheckize %s | filecheck %s --match-full-lines

print("Hello world")

# CHECK: // CHECK: print("Hello world")
