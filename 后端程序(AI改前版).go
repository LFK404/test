package main

import (
	"fmt"
	"math/rand"
	"time"
)

var (
	yanma     string
	lasttime  time.Time
	starttime time.Time
)

func yanzheng(haoma string) bool {
	weishu := len(haoma)
	if weishu != 11 {
		return false
	}
	for _, v := range haoma {
		if v < '0' || v > '9' {
			return false
		}
	}
	return true
}
func youxiao() bool {
	return time.Since(starttime) <= 5*time.Minute
}
func lengque() bool {
	return time.Since(lasttime) <= 60*time.Second
}

const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

func shengcheng() string {
	code := make([]byte, 6)
	for i := range code {
		code[i] = charset[rand.Intn(len(charset))]
	}
	return string(code)
}
func main() {
	rand.Seed(time.Now().UnixNano())
	for {
		fmt.Println("请输入手机号:")
		var shuru string
		fmt.Scanln(&shuru)
		if !yanzheng(shuru) {
			fmt.Println("手机号格式错误")
			continue
		} else {
			fmt.Println("手机号格式正确")
			break
		}
	}
	for {
		fmt.Println("1:输入验证码进行登录,2:获取验证码")
		fmt.Print("请选择:")
		var i int
		fmt.Scanln(&i)
		switch i {
		case 1:
			if yanma == "" {
				fmt.Println("请先获取验证码！")
				continue
			}
			if !youxiao() {
				fmt.Println("验证码已过期，请重新获取")
				yanma = ""
				continue
			}
			fmt.Print("请输入验证码: ")
			var shuma string
			fmt.Scanln(&shuma)
			if shuma == yanma {
				fmt.Println("登录成功")
				return
			} else {
				fmt.Println("验证码错误,请重新选择")
			}
		case 2:
			if lengque() {
				fmt.Println("60秒内只能发送一次，请稍后重试")
				continue
			}
			yanma = shengcheng()
			lasttime = time.Now()
			starttime = time.Now()
			fmt.Printf("验证码已发送：%v\n", yanma)
		default:
			fmt.Println("选择无效，请重新选择")
		}
	}
}
