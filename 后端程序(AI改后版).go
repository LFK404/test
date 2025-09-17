package main

import (
	"fmt"
	"math/rand"
	"time"
)

type PhoneData struct {
	VerificationCode string
	LastRequestTime  time.Time
	GeneratedTime    time.Time
	RequestCount     int
	LastRequestDate  time.Time
}

var phoneDataMap = make(map[string]*PhoneData)
var (
	shuru     string
	yanma     string
	lasttime  time.Time
	starttime time.Time
)

const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

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
func youxiao(phone string) bool {
	data := getOrCreatePhoneData(phone)
	return time.Since(data.GeneratedTime) <= 5*time.Minute
}
func lengque(phone string) bool {
	data := getOrCreatePhoneData(phone)
	return time.Since(data.LastRequestTime) <= 60*time.Second
}

func shengcheng() string {
	code := make([]byte, 6)
	for i := range code {
		code[i] = charset[rand.Intn(len(charset))]
	}
	return string(code)
}
func isSameDay(t1, t2 time.Time) bool {
	return t1.Year() == t2.Year() && t1.Month() == t2.Month() && t1.Day() == t2.Day()
}
func getOrCreatePhoneData(phone string) *PhoneData {
	if data, exists := phoneDataMap[phone]; exists {
		return data
	}
	data := &PhoneData{}
	phoneDataMap[phone] = data
	return data
}
func exceededDailyLimit(phone string) bool {
	data := getOrCreatePhoneData(phone)
	now := time.Now()
	if !isSameDay(data.LastRequestDate, now) {
		data.RequestCount = 0
		data.LastRequestDate = now
	}

	return data.RequestCount >= 5
}
func main() {
	rand.Seed(time.Now().UnixNano())
	for {
		fmt.Println("请输入手机号:")
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
			data := getOrCreatePhoneData(shuru)

			if data.VerificationCode == "" {
				fmt.Println("请先获取验证码！")
				continue
			}

			if !youxiao(shuru) {
				fmt.Println("验证码已过期，请重新获取")
				data.VerificationCode = ""
				continue
			}

			fmt.Print("请输入验证码: ")
			var shuma string
			fmt.Scanln(&shuma)

			if shuma == data.VerificationCode {
				fmt.Println("登录成功")
				return
			} else {
				fmt.Println("验证码错误,请重新选择")
			}
		case 2:
			if lengque(shuru) {
				fmt.Println("60秒内只能发送一次，请稍后重试")
				continue
			}
			if exceededDailyLimit(shuru) {
				fmt.Println("今日验证码发送次数已达上限，请明天再试")
				continue
			}
			data := getOrCreatePhoneData(shuru)
			data.VerificationCode = shengcheng()
			data.LastRequestTime = time.Now()
			data.GeneratedTime = time.Now()
			data.RequestCount++
			fmt.Printf("验证码已发送：%v\n", data.VerificationCode)
		default:
			fmt.Println("选择无效，请重新选择")
		}
	}
}