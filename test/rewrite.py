class Father:
    # 我会写字
    def write(self):
        print("我会写字")


class Son(Father):
    # 重写Son类的write()方法
    # 我会代码
    def write(self):
        print("我会写代码")


son = Son()
son.write()
